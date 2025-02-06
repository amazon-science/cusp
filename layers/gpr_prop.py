import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing

class GPR_prop(MessagePassing):
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, product_manifold=None, euclidean_variant=False, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma
        self.product_manifold = product_manifold
        self.euclidean_variant = euclidean_variant

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0 * torch.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** torch.arange(K + 1, dtype=torch.float)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            TEMP = (alpha) ** torch.arange(K + 1, dtype=torch.float)
            TEMP = TEMP / torch.sum(torch.abs(TEMP))
        elif Init == 'Random':
            bound = torch.sqrt(torch.tensor(3 / (K + 1)))
            TEMP = torch.FloatTensor(K + 1).uniform_(-bound, bound)
            TEMP = TEMP / torch.sum(torch.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = Parameter(TEMP)

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0
        elif self.Init == 'PPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K
        elif self.Init == 'NPPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha ** k
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = torch.sqrt(torch.tensor(3 / (self.K + 1)))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        # Use gcn_norm to normalize adjacency matrix, including edge weights
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype, add_self_loops=True
        )

        # Initialize hidden representation
        if self.euclidean_variant:
            hidden = x * (self.temp[0])
        else:
            hidden = self.product_manifold.mobius_scalar_mul(self.temp[0], x)

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]

            if self.euclidean_variant:
                hidden = hidden + gamma * x
            else:
                gamma_x = self.product_manifold.mobius_scalar_mul(gamma, x)
                hidden = self.product_manifold.mobius_add(hidden, gamma_x)
        return hidden

    def message(self, x_j, norm):
        if self.euclidean_variant:
            return norm.view(-1, 1) * x_j
        else:
            # For manifolds, perform scaling in the tangent space
            tangent_x_j = self.product_manifold.logmap0(x_j)
            scaled_tangent = norm.view(-1, 1) * tangent_x_j
            return self.product_manifold.expmap0(scaled_tangent)

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)