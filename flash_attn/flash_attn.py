import torch


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention2 implementation for 3D inputs (batch, seq_len, d).
        """
        tile_size = 32
        B, N, D = Q.shape

        # Output tensors
        O = torch.zeros_like(Q)
        L = torch.zeros(B, N, device=Q.device, dtype=Q.dtype)

        # Process each batch
        for b in range(B):
            m_i = torch.full((N,), float("-inf"), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(N, device=Q.device, dtype=Q.dtype)
            o_i = torch.zeros(N, D, device=Q.device, dtype=Q.dtype)

            for k0 in range(0, N, tile_size):
                k1 = min(k0 + tile_size, N)
                q = Q[b]  # (N, D)
                k = K[b, k0:k1]  # (tile, D)
                v = V[b, k0:k1]  # (tile, D)

                # Compute attention scores
                attn = torch.matmul(q, k.transpose(0, 1))  # (N, tile)
                attn = attn / (D**0.5)

                # Apply causal masking if needed
                if is_causal:
                    causal_mask = torch.arange(k0, k1, device=Q.device).unsqueeze(0) > torch.arange(
                        N, device=Q.device
                    ).unsqueeze(1)
                    attn = attn.masked_fill(causal_mask, float("-inf"))

                # Update statistics for numerical stability
                m_i_new = torch.maximum(m_i, attn.max(dim=1).values)
                exp_attn = torch.exp(attn - m_i_new[:, None])
                l_i_new = torch.exp(m_i - m_i_new) * l_i + exp_attn.sum(dim=1)
                o_i = (torch.exp(m_i - m_i_new)[:, None] * o_i) + torch.matmul(exp_attn, v)
                m_i = m_i_new
                l_i = l_i_new

            # Normalize output
            o_i = o_i / l_i[:, None]
            O[b] = o_i
            L[b] = m_i + torch.log(l_i)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        # Get dimensions
        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        # Scale factor
        scale = 1.0 / (d**0.5)

        # Compute D = rowsum(dO * O)
        D = (grad_output * O).sum(dim=-1, keepdim=True)  # (batch, n_queries, 1)

        # Recompute attention scores S = Q @ K^T * scale
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch, n_queries, n_keys)

        # Apply causal masking if needed
        if is_causal:
            causal_mask = (
                torch.arange(n_queries, device=Q.device)[:, None] >= torch.arange(n_keys, device=Q.device)[None, :]
            )
            S = torch.where(causal_mask, S, float("-inf"))

        # Recompute P = softmax(S)
        P = torch.softmax(S, dim=-1)  # (batch, n_queries, n_keys)

        # Compute dV = P^T @ dO
        dV = torch.matmul(P.transpose(-2, -1), grad_output)  # (batch, n_keys, d)

        # Compute dP = dO @ V^T
        dP = torch.matmul(grad_output, V.transpose(-2, -1))  # (batch, n_queries, n_keys)

        # Compute dS = P * (dP - D)
        dS = P * (dP - D)  # (batch, n_queries, n_keys)

        # Scale dS
        dS = dS * scale

        # Compute dQ = dS @ K
        dQ = torch.matmul(dS, K)  # (batch, n_queries, d)

        # Compute dK = dS^T @ Q
        dK = torch.matmul(dS.transpose(-2, -1), Q)  # (batch, n_keys, d)

        return dQ, dK, dV, None
