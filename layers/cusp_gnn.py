import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import parse_manifold_config
import geoopt
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

class CuspGNN(nn.Module):
    def __init__(self, input_dim, manifold_config_str, K, alpha, Init, Gamma=None, dropout=0.5, dprate=0.2,
                 use_cusp_laplacian=True, euclidean_variant=False):
        super(CuspGNN, self).__init__()

        self.K = K
        self.alpha = alpha
        self.Init = Init
        self.Gamma = Gamma
        self.dropout = dropout
        self.dprate = dprate
        self.use_cusp_laplacian = use_cusp_laplacian
        self.euclidean_variant = euclidean_variant

        # Parse the manifolds configuration string
        manifolds_config = parse_manifold_config(manifold_config_str)
        self.manifold_dims = [dim for _, dim in manifolds_config]
        # print(self.manifold_dims)

        # Create manifolds and collect dimensions
        manifolds = []
        manifold_list = []
        if euclidean_variant:
            # Single Euclidean manifold
            total_dim = sum(self.manifold_dims)
            manifold = geoopt.Stereographic(k=0.0, learnable=False)
            manifolds.append((manifold, total_dim))
            self.manifold_dims = [total_dim]  # Update manifold_dims
        else:
            for manifold_type, dim in manifolds_config:
                curvature_init = np.random.uniform(0, 1) #If you don't want to randomly init curvature, just set c, k = 1
                # curvature_init = 1
                if manifold_type == 'hyperbolic':
                    # Use PoincareBall with learnable curvature
                    manifold = geoopt.PoincareBall(c=curvature_init, learnable=True)
                elif manifold_type == 'spherical':
                    # Use SphereProjection with learnable curvature
                    manifold = geoopt.SphereProjection(k=curvature_init, learnable=True)
                elif manifold_type == 'euclidean':
                    # Use Stereographic with zero curvature: Learnable is set to false
                    manifold = geoopt.Stereographic(k=0.0, learnable=False)
                else:
                    raise ValueError(f"Unknown manifold type: {manifold_type}")
                manifolds.append((manifold, dim))
                manifold_list.append((manifold, dim))

        # Define the StereographicProductManifold
        self.product_manifold = geoopt.StereographicProductManifold(*manifolds)
        self.manifolds = manifolds

        # Total dimension after concatenation
        self.total_dim = sum(self.manifold_dims)

        # Hidden dimension for intermediate layer
        hidden_dim = 64  # You can set this as a hyperparameter if needed

        # First linear layer to map input features to hidden dimension
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        # Second linear layer to map hidden dimension to manifold dimensions
        self.lin2 = nn.Linear(hidden_dim, self.total_dim)

        # Create a list of GPR_prop instances for each filter
        # Pass the product_manifold to GPR_prop
        # Initialize the propagation module
        self.propagation = ManifoldPropagation(
            K=self.K,
            manifold=self.product_manifold,
            euclidean_variant=self.euclidean_variant
        )

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, data):
        """
        Forward pass for the CuspGNN.

        Args:
            data (dict or Data): If dict, should contain 'x' and 'edge_index'.
                                 If Data, should have 'x' and 'edge_index' attributes.

        Returns:
            Omega_list (list of tensors): List of embeddings from each filter.
        """
        # Handle input data format
        if isinstance(data, dict):
            x = data.get('x', None)
            edge_index = data.get('edge_index', None)
            edge_weight = data.get('edge_weight', None)
        else:
            x = data.x
            edge_index = data.edge_index
            edge_weight = data.edge_weight if self.use_cusp_laplacian and hasattr(data, 'edge_weight') else None

        if x is None or edge_index is None:
            raise ValueError("Input data must contain 'x' and 'edge_index'.")

        num_edges = edge_index.size(1)

        # Assign edge weights
        if self.use_cusp_laplacian and edge_weight is not None:
            edge_weight = edge_weight  
        else:
            # Assign edge weights as ones to recover standard graph Laplacian
            edge_weight = torch.ones(num_edges, dtype=torch.float, device=edge_index.device)

        # Apply dropout and first linear layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)  # Shape: (N, total_dim)

        # Apply dprate dropout
        if self.dprate > 0:
            x = F.dropout(x, p=self.dprate, training=self.training)

        # Initialize embeddings in the product manifold using exponential map
        if self.euclidean_variant:
            x_manifold = x  # In Euclidean space, expmap0 is identity
        else:
            x_manifold = self.product_manifold.expmap0(x)  # Shape: (N, total_dim), on the manifold

        # Perform manifold-aware propagation and collect features at each step
        Omega_list = self.propagation(x_manifold, edge_index, edge_weight)

        return Omega_list



class ManifoldPropagation(MessagePassing):
    def __init__(self, K, manifold, euclidean_variant=False, **kwargs):
        super(ManifoldPropagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.manifold = manifold
        self.euclidean_variant = euclidean_variant

    def forward(self, x, edge_index, edge_weight=None):
        # Compute normalized edge weights
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype, add_self_loops=True
        )

        h = x  # Initial features
        Omega_list = [h]  # List to store features at each step

        for k in range(1, self.K + 1):
            h = self.propagate(edge_index, x=h, norm=norm)
            Omega_list.append(h)

        return Omega_list

    def message(self, x_j, norm):
        if self.euclidean_variant:
            return norm.view(-1, 1) * x_j
        else:
            # Manifold-aware message passing
            # Map x_j to tangent space
            tangent_x_j = self.manifold.logmap0(x_j)
            # Scale by norm
            scaled_tangent = norm.view(-1, 1) * tangent_x_j
            # Map back to manifold
            return self.manifold.expmap0(scaled_tangent)