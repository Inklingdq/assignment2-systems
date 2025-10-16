import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L pointer - for logsumexp values
    L_offset = batch_index * stride_lb + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # Load Q tile
    Q = tl.load(Q_block_ptr)

    # Initialize O, l, m in float32 for numerical stability
    O = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)

    # Number of key tiles
    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Loop over key tiles
    for key_tile_idx in range(n_key_tiles):
        # Load K and V tiles
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)

        # Compute attention scores: S = Q @ K^T / sqrt(d)
        # Q: [Q_TILE_SIZE, D], K: [D, K_TILE_SIZE]
        S = tl.dot(Q, K) * scale  # [Q_TILE_SIZE, K_TILE_SIZE]

        # Apply causal masking if needed
        if IS_CAUSAL:
            # Get query and key indices
            query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_indices = key_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # Create causal mask: query_idx >= key_idx
            causal_mask = query_indices[:, None] >= key_indices[None, :]
            S = tl.where(causal_mask, S, -1e6)

        # Update m: m_new = max(m_old, rowmax(S))
        m_new = tl.maximum(m, tl.max(S, axis=1))

        # Compute P_tilde = exp(S - m_new)
        P_tilde = tl.exp(S - m_new[:, None])

        # Update l: l_new = exp(m_old - m_new) * l_old + rowsum(P_tilde)
        l_new = tl.exp(m - m_new) * l + tl.sum(P_tilde, axis=1)

        # Update O: O_new = diag(exp(m_old - m_new)) * O_old + P_tilde @ V
        # Cast P_tilde to V's dtype before multiplication
        P_tilde_casted = P_tilde.to(V.dtype)
        O_scaled = tl.exp(m - m_new)[:, None] * O
        O = O_scaled + tl.dot(P_tilde_casted, V, acc=tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32))

        # Update m and l
        m = m_new
        l = l_new

        # Advance block pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Normalize O: O = O / l
    O = O / l[:, None]

    # Compute L (logsumexp): L = m + log(l)
    L = m + tl.log(l)

    # Cast O to the appropriate dtype before writing
    O_casted = O.to(O_block_ptr.type.element_ty)

    # Write O and L to global memory
    tl.store(O_block_ptr, O_casted)
    tl.store(L_ptr + L_offset, L, mask=L_offset < (batch_index + 1) * stride_lb)


@torch.compile
def flash_attention_backward(Q, K, V, O, dO, L, is_causal, scale):
    """
    FlashAttention backward pass with recomputation.

    Args:
        Q: Query tensor of shape (batch, n_queries, d)
        K: Key tensor of shape (batch, n_keys, d)
        V: Value tensor of shape (batch, n_keys, d)
        O: Output tensor of shape (batch, n_queries, d)
        dO: Gradient of output of shape (batch, n_queries, d)
        L: Log-sum-exp of shape (batch, n_queries)
        is_causal: Whether causal masking was applied
        scale: Scale factor (1/sqrt(d))

    Returns:
        dQ, dK, dV: Gradients with respect to Q, K, V
    """
    # Get dimensions
    batch_size, n_queries, d = Q.shape
    _, n_keys, _ = K.shape

    # Compute D = rowsum(dO * O)
    D = (dO * O).sum(dim=-1, keepdim=True)  # (batch, n_queries, 1)

    # Recompute attention scores S = Q @ K^T * scale
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, n_queries, n_keys)

    # Apply causal masking if needed
    if is_causal:
        causal_mask = (
            torch.arange(n_queries, device=Q.device)[:, None] >= torch.arange(n_keys, device=Q.device)[None, :]
        )
        S = torch.where(causal_mask, S, -1e6)

    # Recompute P = softmax(S)
    P = torch.softmax(S, dim=-1)  # (batch, n_queries, n_keys)

    # Compute dV = P^T @ dO
    dV = torch.matmul(P.transpose(-2, -1), dO)  # (batch, n_keys, d)

    # Compute dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-2, -1))  # (batch, n_queries, n_keys)

    # Compute dS = P * (dP - D)
    dS = P * (dP - D)  # (batch, n_queries, n_keys)

    # Scale dS
    dS = dS * scale

    # Compute dQ = dS @ K
    dQ = torch.matmul(dS, K)  # (batch, n_queries, d)

    # Compute dK = dS^T @ Q
    dK = torch.matmul(dS.transpose(-2, -1), Q)  # (batch, n_keys, d)

    return dQ, dK, dV


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention2 implementation using Triton kernels.

        Args:
            Q: Query tensor of shape (batch, n_queries, d)
            K: Key tensor of shape (batch, n_keys, d)
            V: Value tensor of shape (batch, n_keys, d)
            is_causal: Whether to apply causal masking

        Returns:
            O: Output tensor of shape (batch, n_queries, d)
        """
        # Get shapes
        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        # Ensure inputs are contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        # Allocate output tensors
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

        # Tile sizes
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        # Scale factor
        scale = 1.0 / (d**0.5)

        # Launch grid: (Tq, batch_size)
        n_query_tiles = triton.cdiv(n_queries, Q_TILE_SIZE)
        grid = (n_query_tiles, batch_size)

        # Launch kernel
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )

        # Save for backward
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale

        dQ, dK, dV = flash_attention_backward(Q, K, V, O, grad_output, L, is_causal, scale)

        return dQ, dK, dV, None
