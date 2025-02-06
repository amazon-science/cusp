import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.cusp_gnn import CuspGNN
from layers.curvature_encoding import CurvatureEncoding
from layers.cusp_pooling import CuspPooling
from utils.data_utils import parse_manifold_config
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class CUSPModel(nn.Module):
    def __init__(self, input_dim, output_dim, manifold_config_str, K, alpha, Init, Gamma, d_f,
                 num_frequencies=16, dropout=0.5, dprate=0.0,
                 use_curvature_encoding=True, use_cusp_pooling=True, euclidean_variant=False,
                 use_cusp_laplacian=True):
        super(CUSPModel, self).__init__()

        self.use_curvature_encoding = use_curvature_encoding
        self.use_cusp_pooling = use_cusp_pooling
        

        # Adjust manifold configuration for Euclidean variant
        if euclidean_variant:
            manifolds_config = parse_manifold_config(manifold_config_str)
            total_dim = sum([dim for _, dim in manifolds_config])
            manifold_config_str = f"E{total_dim}"

        # Initialize the CuspGNN with filter bank
        self.gprgnn = CuspGNN(
            input_dim=input_dim,
            manifold_config_str=manifold_config_str,
            K=K,
            alpha=alpha,
            Init=Init,
            Gamma=Gamma,
            dropout=dropout,
            dprate=dprate,
            use_cusp_laplacian=use_cusp_laplacian,
            euclidean_variant=euclidean_variant
        )

        # Retrieve the product_manifold from gprgnn
        self.product_manifold = self.gprgnn.product_manifold

        # Number of filters
        self.num_filters = K+1

        manifold_dim = sum(self.gprgnn.manifold_dims)

        # Initialize Curvature Encoding if used
        if self.use_curvature_encoding:
            self.curvature_encoding = CurvatureEncoding(
                d_f=d_f,
                manifolds=self.gprgnn.manifolds,
                manifold_dim = manifold_dim,
                num_frequencies=num_frequencies,
                product_manifold=self.product_manifold,
                euclidean_variant=euclidean_variant
            )
        else:
            self.curvature_encoding = None

        # Initialize Cusp Pooling with hierarchical attention if used
        if self.use_cusp_pooling:
            self.cusp_pooling = CuspPooling(
                manifold_dims=self.gprgnn.manifold_dims,
                manifolds=self.gprgnn.manifolds,
                num_components=len(self.gprgnn.manifold_dims),
                num_filters=self.num_filters,
                d_f=d_f,
                use_curvature_encoding=self.use_curvature_encoding,
                euclidean_variant=euclidean_variant
            )
            # Adjust total_dim for the final output layer
            total_dim = manifold_dim
            # print(total_dim)
            if self.use_curvature_encoding:
                total_dim += d_f
            # print(total_dim)
        else:
            total_dim = manifold_dim * self.num_filters  # Adjusted for concatenation over filters
            if self.use_curvature_encoding:
                total_dim += d_f

        self.output_layer = nn.Linear(total_dim, output_dim)

    def forward(self, data):
        # Get the filter bank outputs (list of tensors)
        Omega_list = self.gprgnn(data)

        # Initialize curvature_embeddings
        curvature_embeddings = None

        if self.use_curvature_encoding and self.curvature_encoding is not None:
            # Compute curvature embeddings
            kappa = data.kappa if hasattr(data, 'kappa') else None
            curvature_embeddings = self.curvature_encoding(kappa)

        if self.use_cusp_pooling and self.cusp_pooling is not None:
            final_embedding = self.cusp_pooling(Omega_list, curvature_embeddings)
        else:
            # Simple embedding concatenation over filters
            final_embedding = torch.cat(Omega_list, dim=1)
            if curvature_embeddings is not None:
                # Concatenate curvature embedding once
                final_embedding = torch.cat([final_embedding, curvature_embeddings], dim=1)

        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        # Final output layer
        output = self.output_layer(final_embedding)

        return F.log_softmax(output, dim=1)


    def encode(self, x, edge_index, kappa = None):
        """
        Generates node embeddings.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            curvature_embeddings (Tensor, optional): Curvature embeddings (N, d_f).

        Returns:
            Tensor: Node embeddings.
        """
        data = {'x': x, 'edge_index': edge_index, 'kappa': kappa}
        Omega_list = self.gprgnn(data)  # List of embeddings from each filter

        if self.use_cusp_pooling and self.use_curvature_encoding:
            if kappa is None:
                raise ValueError("Curvature embeddings are required for Cusp Pooling.")
            curvature_embeddings = self.curvature_encoding(kappa)  # Shape: (N, d_f)
            z = self.cusp_pooling(Omega_list, curvature_embeddings)
        else:
            # Simple concatenation of filter outputs
            z = torch.cat(Omega_list, dim=1)  # Shape: (N, total_dim)

        # Normalize embeddings to unit norm
        z = F.normalize(z, p=2, dim=1)

        return z  # Return node embeddings


    # Link Prediction specific methods
    def decode(self, z, edge_index):
        """
        Inner product decoder for Link Prediction.

        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Edge indices.

        Returns:
            Tensor: Scores for each edge.
        """
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def link_prediction_loss(self, z, pos_edge_index, neg_edge_index):
        """
        Computes the loss for Link Prediction.

        Args:
            z (Tensor): Node embeddings.
            pos_edge_index (Tensor): Positive edge indices.
            neg_edge_index (Tensor): Negative edge indices.

        Returns:
            Tensor: Loss value.
        """
        pos_scores = self.decoder(z, pos_edge_index)
        neg_scores = self.decoder(z, neg_edge_index)

        pos_labels = torch.ones(pos_scores.size(0), device=z.device)
        neg_labels = torch.zeros(neg_scores.size(0), device=z.device)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def test(self, z, pos_edge_index, neg_edge_index):
        """
        Evaluates the model on Link Prediction metrics.

        Args:
            z (Tensor): Node embeddings.
            pos_edge_index (Tensor): Positive edge indices.
            neg_edge_index (Tensor): Negative edge indices.

        Returns:
            Tuple[float, float]: AUC and AP scores.
        """
        pos_scores = self.decode(z, pos_edge_index).detach().cpu().numpy()
        neg_scores = self.decode(z, neg_edge_index).detach().cpu().numpy()

        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        return auc, ap

    def get_curvatures(self):
        """
        Returns a dictionary of learned curvatures for each manifold component.
        """
        curvatures = {}
        for idx, (manifold, _) in enumerate(self.gprgnn.manifolds):
            if hasattr(manifold, 'c'):
                # For PoincareBall manifold
                curvatures[f'manifold_{idx}'] = manifold.k.item()
            elif hasattr(manifold, 'k'):
                # For Sphere manifold
                curvatures[f'manifold_{idx}'] = manifold.k.item()
            else:
                # For Euclidean manifold (curvature is zero)
                curvatures[f'manifold_{idx}'] = 0.0
        return curvatures

    def get_filter_weights(self):
        """
        Returns the filter attention weights (epsilon) after softmax.
        """
        if hasattr(self, 'cusp_pooling') and self.cusp_pooling is not None:
            epsilon = F.softmax(self.cusp_pooling.epsilon, dim=0).detach().cpu().numpy()
            return epsilon
        else:
            return None

    def get_component_weights(self):
        """
        Returns the manifold component attention parameters (theta).
        """
        if hasattr(self, 'cusp_pooling') and self.cusp_pooling is not None:
            theta_values = [theta.detach().cpu().numpy() for theta in self.cusp_pooling.theta]
            return theta_values
        else:
            return None