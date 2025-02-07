import torch
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci

class CuspLaplacian:
    def __init__(self, nx_graph, num_nodes, alpha=0.5):
        """
        Compute the  Cusp aplacian and curvature values.

        Args:
            nx_graph: NetworkX graph.
            alpha: Parameter for Ollivier-Ricci curvature.
        """
        self.nx_graph = nx_graph
        self.alpha = alpha
        self.num_nodes = num_nodes
        # Add self-loops to the graph
        self_loops = [(n, n) for n in nx_graph.nodes()]
        nx_graph.add_edges_from(self_loops)
        # print(self.nx_graph.nodes())

        # Compute Ollivier-Ricci curvature
        self.compute_ricci_curvature()

    def compute_ricci_curvature(self):
        # Compute Ricci curvature using GraphRicciCurvature package
        orc = OllivierRicci(self.nx_graph, alpha=self.alpha, verbose="ERROR")
        orc.compute_ricci_curvature()
        self.G_orc = orc.G.copy()
        # print(self.G_orc.nodes())

    def get_ricci_edge_weights(self, edge_index):
        """
        Returns the Ricci edge weights as a torch tensor,
        matching the order of edges in edge_index.

        Args:
            edge_index: Tensor of shape (2, E), the edge indices from data.edge_index.

        Returns:
            edge_weights: Tensor of shape (E,), curvature-based weights for edges.
        """
        edge_weights = []

        # Convert edge_index to list of tuples
        edge_list = edge_index.t().tolist()  # Shape: (E, 2)

        for u, v in edge_list:
            # Get Ollivier-Ricci curvature for edge (u, v)
            if self.G_orc.has_edge(u, v):
                edge_data = self.G_orc[u][v]
            elif self.G_orc.has_edge(v, u):
                edge_data = self.G_orc[v][u]
            else:
                # Edge might have been removed; assign default curvature
                edge_data = {'ricciCurvature': 0.0}

            kappa = edge_data.get('ricciCurvature', 0.0)
            # Compute curvature-based weight
            if kappa == 1.0:
                # Avoid division by zero when kappa == 1
                weight = np.exp(-1e6)
            else:
                weight = np.exp(-1 / (1 - kappa))
            edge_weights.append(weight)

        return torch.tensor(edge_weights, dtype=torch.float)

    def get_curvature_values(self):
        """
        Returns the curvature values for nodes.

        Returns:
            node_curvature: Tensor of shape (N,), Ricci curvature values for nodes.
        """
        node_curvature = torch.zeros(self.num_nodes, dtype=torch.float)

        for n, data in self.G_orc.nodes(data=True):
            node_curvature[n] = data.get('ricciCurvature', 0.0)
        return node_curvature