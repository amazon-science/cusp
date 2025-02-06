import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CurvatureEncoding(nn.Module):
    def __init__(self, d_f, manifolds, manifold_dim, num_frequencies=16, method='mlp_res', product_manifold=None, euclidean_variant=False):
        """
        Curvature Encoding layer using inverse CDF transformation.

        Args:
            d_f: Dimensionality of output features.
            manifolds: List of tuples (manifold, dimension).
            num_frequencies: Number of frequencies to use in the encoding.
            method: Method for inverse CDF ('mlp_res', 'maf', 'iaf', 'NVP').
            product_manifold: Product manifold instance.
            euclidean_variant: Boolean flag indicating if Euclidean variant is used.
        """
        super(CurvatureEncoding, self).__init__()
        self.d_f = d_f
        self.manifolds = manifolds
        self.num_frequencies = num_frequencies
        self.method = method
        self.euclidean_variant = euclidean_variant
        self.manifold_dim = manifold_dim
        self.product_manifold = product_manifold

        # Define MLP layers for 'mlp_res' method
        if method == 'mlp_res':
            self.mlp1 = nn.Linear(self.num_frequencies, self.num_frequencies)
            self.mlp2 = nn.Linear(self.num_frequencies, self.num_frequencies)
            self.mlp3 = nn.Linear(self.num_frequencies, self.num_frequencies)

        # Linear layer to project the embeddings to dimension d_f
        self.projection1 = nn.Linear(2 * num_frequencies, self.manifold_dim)
        self.projection2 = nn.Linear(self.manifold_dim, d_f)

    def generate_frequencies(self):
        """
        Generates frequencies using inverse CDF transformation.

        Returns:
            sampled_freq: Tensor of shape (1, num_frequencies)
        """
        if self.method == 'mlp_res':
            # Sample frequencies from uniform distribution
            sampled_freq = torch.rand(1, self.num_frequencies, device=self.projection1.weight.device)
            # Apply transformation: sampled_freq = 1 / 10 ** sampled_freq
            sampled_freq = 1 / (10 ** sampled_freq)

            # Pass through MLP layers with residual connections
            sampled_freq1 = F.relu(self.mlp1(sampled_freq))
            sampled_freq2 = self.mlp2(sampled_freq1)
            sampled_freq = self.mlp3(sampled_freq2 + sampled_freq)
        else:
            raise ValueError(f"Method '{self.method}' not implemented in CurvatureEncoding.")

        return sampled_freq  # Shape: (1, num_frequencies)

    def forward(self, kappa):
        """
        Args:
            kappa: Curvature values for nodes (N,)
        Returns:
            curvature_embeddings: Tensor of shape (N, d_f)
        """
        # Handle missing kappa
        if kappa is None:
            kappa = torch.zeros(1, device=self.projection.weight.device)

        N = kappa.shape[0]

        # Generate frequencies
        sampled_freq = self.generate_frequencies()  # Shape: (1, num_frequencies)

        # Expand sampled_freq to match batch size
        sampled_freq = sampled_freq.expand(N, self.num_frequencies)  # Shape: (N, num_frequencies)

        # Compute embeddings
        kappa = kappa.unsqueeze(1)  # Shape: (N, 1)
        embeddings = torch.cat([
            torch.sin(kappa * sampled_freq),
            torch.cos(kappa * sampled_freq)
        ], dim=-1)  # Shape: (N, 2 * num_frequencies)

        # Scale embeddings
        embeddings = embeddings * np.sqrt(1 / self.d_f)

        # Project to desired dimension
        embeddings = self.projection1(embeddings)  # Shape: (N, d_f)

        # print(embeddings.shape)

        # Map embeddings into the manifold if needed
        if self.euclidean_variant:
            curvature_embeddings = embeddings  # No mapping needed
        else:
            # Map embeddings into the product manifold
            curvature_embeddings = self.product_manifold.expmap0(embeddings)

        # Project to desired dimension
        curvature_embeddings = self.projection2(curvature_embeddings)  # Shape: (N, d_f)

        return curvature_embeddings