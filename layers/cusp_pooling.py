import torch
import torch.nn as nn
import torch.nn.functional as F

class CuspPooling(nn.Module):
    def __init__(self, manifold_dims, manifolds, num_components, num_filters, d_f,
                 use_curvature_encoding=True, euclidean_variant=False):
        super(CuspPooling, self).__init__()
        self.num_components = num_components  # Q
        self.num_filters = num_filters        # L
        self.d_f = d_f
        self.manifold_dims = manifold_dims
        self.manifolds = manifolds  # List of (manifold, dim)
        self.use_curvature_encoding = use_curvature_encoding
        self.euclidean_variant = euclidean_variant

        # Weight matrices W_q for Möbius linear transformations
        self.W_q = nn.ModuleList()
        for q in range(num_components):
            dim = manifold_dims[q]
            self.W_q.append(nn.Linear(dim, dim))

        # θ_q parameters for component attention (one per component)
        self.theta = nn.ParameterList()
        for q in range(num_components):
            dim_q = manifold_dims[q]
            self.theta.append(nn.Parameter(torch.randn(dim_q)))

        # Learnable parameters ε_l for filter attention
        self.epsilon = nn.Parameter(torch.randn(num_filters))

    def forward(self, Omega_list, curvature_embeddings):
        N = Omega_list[0].shape[0]  # Number of nodes

        # Initialize lists to store embeddings for each filter
        zeta_l_list = []

        for l in range(self.num_filters):
            # Get the output from filter l
            Omega_l = Omega_list[l]  # Shape: (N, total_dim)

            # Split Omega_l into components
            Z_q_list = torch.split(Omega_l, self.manifold_dims, dim=1)

            # Initialize lists to store per-component results
            log_mapped_list = []
            WZ_list = []
            for q in range(self.num_components):
                Z_q = Z_q_list[q]  # Embedding for component q, shape: (N, dim_q)
                manifold = self.manifolds[q][0]
                dim_q = self.manifold_dims[q]
                W_q = self.W_q[q]

                # Möbius linear transformation: W_q ⊗_{κ_q} Z_q
                if self.euclidean_variant:
                    WZ_q = W_q(Z_q)  # Standard linear transformation
                else:
                    # Compute log_map at zero
                    log_z = manifold.logmap0(Z_q)  # Shape: (N, dim_q)
                    # Apply linear transformation in the tangent space
                    W_log_z = W_q(log_z)  # Shape: (N, dim_q)
                    # Map back to the manifold using exp_map at zero
                    WZ_q = manifold.expmap0(W_log_z)  # Shape: (N, dim_q)

                # Map to tangent space using log map at zero: log_{0}^{κ_q}(WZ_q)
                if self.euclidean_variant:
                    log_mapped_q = WZ_q  # In Euclidean space, logmap0 is identity
                else:
                    log_mapped_q = manifold.logmap0(WZ_q)  # Shape: (N, dim_q)

                log_mapped_list.append(log_mapped_q)
                WZ_list.append(WZ_q)

            # Compute centroid μ^{(L)}
            combined_embeddings = torch.cat(log_mapped_list, dim=1)  # Shape: (N, total_dim)
            mu_L = combined_embeddings.mean(dim=0, keepdim=True)  # Shape: (1, total_dim)

            # Compute attention scores α_q and attention weights β_q
            alpha_q_list = []
            start_idx = 0
            for q in range(self.num_components):
                dim_q = log_mapped_list[q].shape[1]
                end_idx = start_idx + dim_q

                combined_embedding = log_mapped_list[q]  # Shape: (N, dim_q)
                mu_L_q = mu_L[:, start_idx:end_idx]  # Shape: (1, dim_q)
                diff = combined_embedding - mu_L_q  # Shape: (N, dim_q)

                # Compute α_q = σ( θ_q^T (diff) )
                theta_q = self.theta[q]  # Shape: (dim_q,)
                alpha_q = torch.sigmoid(torch.matmul(diff, theta_q))  # Shape: (N,)
                alpha_q_list.append(alpha_q)

                start_idx = end_idx

            # Stack attention scores and compute attention weights β_q
            alpha_q_tensor = torch.stack(alpha_q_list, dim=0)  # Shape: (Q, N)
            beta_q = F.softmax(alpha_q_tensor, dim=0)  # Shape: (Q, N)

            # Fuse the components using attention weights β_q
            zeta_components = []
            for q in range(self.num_components):
                beta_q_expanded = beta_q[q].unsqueeze(-1)  # Shape: (N, 1)
                WZ_q = WZ_list[q]  # Shape: (N, dim_q)
                manifold = self.manifolds[q][0]

                # Scale the embeddings by β_q in the manifold: β_q ⊗_{κ_q} WZ_q
                if self.euclidean_variant:
                    scaled_embedding = beta_q_expanded * WZ_q  # Standard scalar multiplication
                else:
                    scaled_embedding = manifold.mobius_scalar_mul(beta_q_expanded, WZ_q)

                zeta_components.append(scaled_embedding)

            # Concatenate fused components along last dimension
            zeta_l = torch.cat(zeta_components, dim=1)  # Shape: (N, total_dim)

            zeta_l_list.append(zeta_l)

        # Compute attention weights ε_l over filters
        epsilon = F.softmax(self.epsilon, dim=0)  # Shape: (num_filters,)

        # Aggregate over filters using ε_l
        zeta_l_tensor = torch.stack(zeta_l_list, dim=0)  # Shape: (num_filters, N, final_dim)
        epsilon_expanded = epsilon.view(self.num_filters, 1, 1)  # Shape: (num_filters, 1, 1)
        zeta = torch.sum(epsilon_expanded * zeta_l_tensor, dim=0)  # Shape: (N, final_dim)

        # Concatenate curvature embedding once to the final fused embedding
        if self.use_curvature_encoding and curvature_embeddings is not None:
            zeta = torch.cat([zeta, curvature_embeddings], dim=1)  # Shape: (N, final_dim + d_f)

        # print("zeta", zeta.shape)
        return zeta  # Final embedding