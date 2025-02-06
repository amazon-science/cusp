import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.utils import train_test_split_edges, to_undirected
from models.cusp_model import CUSPModel
from layers.cusp_laplacian import CuspLaplacian
import networkx as nx
import numpy as np
import random
import geoopt
import torch_geometric.transforms as T
from sklearn.metrics import f1_score
# import wandb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="CUSP Model Training with Node Classification and Link Prediction")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of experiment runs')  # Added num_runs
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'PubMed', 'Chameleon', 'Actor', 'Squirrel', 'Texas', 'Cornell'], help='Dataset name')
    parser.add_argument('--model', type=str, default='cusp', choices=['cusp', 'gcn', 'gat', 'sage', 'gprgnn'], help='Model to use (CUSP, GCN, GAT, GraphSAGE, or GPRGNN)')
    parser.add_argument('--manifold_config', type=str, default='H16H16S16E16', help='Manifold configuration string')
    parser.add_argument('--K', type=int, default=10, help='Maximum number of propagation steps')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter for GPR propagation')
    parser.add_argument('--Init', type=str, default='PPR', choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS'], help='Initialization method for GPR weights')
    parser.add_argument('--Gamma', type=float, default=None, help='Gamma parameter for GPR weights')
    parser.add_argument('--d_f', type=int, default=64, help='Dimensionality of curvature embeddings per component')
    parser.add_argument('--num_frequencies', type=int, default=16, help='Number of frequencies for curvature encoding')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dprate', type=float, default=0.5, help='Dropout rate for propagation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'radam'], help='Optimizer to use')
    parser.add_argument('--ricci_alpha', type=float, default=0.5, help='Alpha parameter for Ollivier-Ricci curvature')
    parser.add_argument('--task', type=str, default='node_classification', choices=['node_classification', 'link_prediction'], help='Task to perform')
    parser.add_argument('--use_cusp_laplacian', action='store_true', help='Use Cusp Laplacian (default). If not set, uses standard graph Laplacian.')
    parser.add_argument('--use_curvature_encoding', action='store_true', help='Use curvature-based positional encoding in Cusp Pooling.')
    parser.add_argument('--use_cusp_pooling', action='store_true', help='Use Cusp Pooling with hierarchical attention. If not set, uses simple embedding concatenation.')
    parser.add_argument('--euclidean_variant', action='store_true', help='Use Euclidean variant of the model (all manifolds are Euclidean).')
    parser.add_argument('--wandb_project', type=str, default='CUSP_GNN', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, help='WandB entity (team/user)')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--ppnp', type=str, default='GPR_prop', choices=['PPNP', 'GPR_prop'], help='Propagation method')
    args = parser.parse_args()

    # Initialize WandB
    # wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Collect metrics over runs
    metrics = []

    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")
        # Set random seed for this run
        current_seed = args.seed + run
        set_seed(current_seed)

        # Load all datasets (reload for each run to ensure randomness in splits)
        # Downloads all together in the first run
        datasets = {
            "Cora": Planetoid(root="data/Cora", name="Cora", transform=T.ToUndirected()),
            "Citeseer": Planetoid(root="data/Citeseer", name="Citeseer", transform=T.ToUndirected()),
            "PubMed": Planetoid(root="data/PubMed", name="PubMed", transform=T.ToUndirected()),
            "Chameleon": WikipediaNetwork(root="data/WikipediaNetwork", name="chameleon", transform=T.ToUndirected()),
            "Actor": Actor(root="data/Actor", transform=T.ToUndirected()),
            "Squirrel": WikipediaNetwork(root="data/WikipediaNetwork", name="squirrel", transform=T.ToUndirected()),
            "Texas": WebKB(root="data/WebKB", name="Texas", transform=T.ToUndirected()),
            "Cornell": WebKB(root="data/WebKB", name="Cornell", transform=T.ToUndirected())
        }

        dataset = datasets.get(args.dataset)
        if dataset is None:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    
        data = dataset[0]
        if args.dataset in ['Chameleon', 'Actor', 'Squirrel', 'Texas', 'Cornell']:
            #Because there are multiple masks present in these datasets, and we use just one
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]

        num_nodes = data.x.shape[0]

        # Set model input and output dimensions
        input_dim = data.num_features
        output_dim = dataset.num_classes

        # Convert edge_index to NetworkX graph
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        edge_list = edge_index.t().tolist()  # Shape: (E, 2)

        G = nx.Graph()
        G.add_edges_from(edge_list)


        if args.use_cusp_laplacian:
            cusp_laplacian = CuspLaplacian(nx_graph=G, num_nodes = num_nodes, alpha=args.ricci_alpha)
            data.edge_weight = cusp_laplacian.get_ricci_edge_weights(data.edge_index)
            data.kappa = cusp_laplacian.get_curvature_values()  # Curvature values for nodes (N,)
        else:
            # Assign edge weights as ones to recover standard graph Laplacian
            num_edges = data.edge_index.size(1)
            data.edge_weight = torch.ones(num_edges, dtype=torch.float, device=data.edge_index.device)
            data.kappa = torch.zeros(data.num_nodes, dtype=torch.float, device=data.edge_index.device)  # All curvatures are 0 in Euclidean space

        # If task is link prediction, split edges
        if args.task == 'link_prediction':
            # Preserve node features before splitting edges
            x = data.x.clone()

            # Ensure the graph is undirected
            data.edge_index = to_undirected(data.edge_index)

            # Split edges into train/val/test sets
            data = train_test_split_edges(data)

            # Restore node features
            data.x = x

        # Define model based on the selected argument
        if args.model == 'cusp':
            model = CUSPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                manifold_config_str=args.manifold_config,
                K=args.K,
                alpha=args.alpha,
                Init=args.Init,
                Gamma=args.Gamma,
                d_f=args.d_f,
                num_frequencies=args.num_frequencies,
                dropout=args.dropout,
                dprate=args.dprate,
                use_curvature_encoding=args.use_curvature_encoding,
                use_cusp_pooling=args.use_cusp_pooling,
                euclidean_variant=args.euclidean_variant,
                use_cusp_laplacian=args.use_cusp_laplacian
            )
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        model = model.to(device)
        data = data.to(device)

        # Define optimizer (Choose between Riemannian Adam and the traditional Adam operator)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'radam':
            optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr, stabilize=10)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # Define loss functions based on task
        if args.task == 'node_classification':
            criterion = F.nll_loss
            # Training loop
            best_metric = train_node_classification(model, data, optimizer, scheduler, args)
        elif args.task == 'link_prediction':
            criterion = F.binary_cross_entropy_with_logits
            # Training loop
            best_metric = train_link_prediction(model, data, optimizer, scheduler, args)
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        metrics.append(best_metric)

    # Compute average and standard deviation of best metrics
    avg_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    print(f"\nFinal Results over {args.num_runs} runs:")
    if args.task == 'node_classification':
        print(f"Best Test F1 Score: {avg_metric:.4f} ± {std_metric:.4f}")
    elif args.task == 'link_prediction':
        aucs, aps = zip(*metrics)
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        avg_ap = np.mean(aps)
        std_ap = np.std(aps)
        print(f"Best AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"Best AP: {avg_ap:.4f} ± {std_ap:.4f}")

def train_node_classification(model, data, optimizer, scheduler, args):
    """
    Training loop for Node Classification.
    Returns the best test F1 score.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    best_val_f1 = 0
    best_test_f1 = 0

    # Lists to store metrics over epochs
    train_f1_list = []
    val_f1_list = []
    test_f1_list = []
    loss_list = []
    curvature_logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)  # Raw logits
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_f1, val_f1, test_f1 = evaluate_node_classification(model, data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_f1 = test_f1

        # Store metrics
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)
        test_f1_list.append(test_f1)
        loss_list.append(loss.item())

        # Log metrics to WandB
        # wandb.log({
        #     'epoch': epoch,
        #     'loss': loss.item(),
        #     'train_f1': train_f1,
        #     'val_f1': val_f1,
        #     'test_f1': test_f1
        # })

        # Get curvatures and log them
        curvatures = model.get_curvatures()
        curvature_logs.append(curvatures)
        # for key, value in curvatures.items():
        #     wandb.log({f'curvature/{key}': value, 'epoch': epoch})

        # if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, '
              f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')

    print(f'Best Val F1: {best_val_f1:.4f}, Best Test F1: {best_test_f1:.4f}')

    # After training, print filter weights, component weights, and curvatures
    filter_weights = model.get_filter_weights()
    if filter_weights is not None:
        print('\nFilter Weights (epsilon):')
        print(filter_weights)
        # wandb.log({'filter_weights': wandb.Histogram(filter_weights)})

    component_weights = model.get_component_weights()
    if component_weights is not None:
        print('\nComponent Weights (theta):')
        for idx, theta in enumerate(component_weights):
            print(f'Theta {idx}: {theta}')
            # wandb.log({f'component_weights/theta_{idx}': wandb.Histogram(theta)})

    print('\nLearned Curvatures:')
    final_curvatures = model.get_curvatures()
    for key, value in final_curvatures.items():
        print(f'{key}: {value}')
        # wandb.log({f'final_curvature/{key}': value})

    # Report best metrics to WandB
    # wandb.log({'best_val_f1': best_val_f1, 'best_test_f1': best_test_f1})

    return best_test_f1  # Return the best test F1 score

def evaluate_node_classification(model, data):
    """
    Evaluation function for Node Classification, returning F1 score instead of accuracy.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1).cpu().numpy()  # Convert predictions to numpy
        labels = data.y.cpu().numpy()  # Convert true labels to numpy

        # Calculate F1 score for train, validation, and test sets
        train_f1 = f1_score(labels[data.train_mask.cpu()], preds[data.train_mask.cpu()], average='weighted')
        val_f1 = f1_score(labels[data.val_mask.cpu()], preds[data.val_mask.cpu()], average='weighted')
        test_f1 = f1_score(labels[data.test_mask.cpu()], preds[data.test_mask.cpu()], average='weighted')

    return train_f1, val_f1, test_f1



# Modify train_link_prediction to return best AUC and AP
def train_link_prediction(model, data, optimizer, scheduler, args):
    """
    Training loop for Link Prediction, handling both CUSP and baseline models.
    Returns the best AUC and AP scores.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    best_auc = 0
    best_ap = 0

    # Lists to store metrics over epochs
    auc_list = []
    ap_list = []
    loss_list = []
    curvature_logs = []

    train_neg_edge_index = sample_neg_edges_from_mask(data.train_neg_adj_mask, num_neg_edges=data.train_pos_edge_index.size(1))

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Generate node embeddings using training edges
        if args.model == 'cusp':
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        else:
            # Baseline models (GCN, GAT, SAGE) don't use kappa
            z = model.encode(data.x, data.train_pos_edge_index)

        # Compute loss using positive and negative edges
        loss = link_prediction_loss(model, z, data.train_pos_edge_index, train_neg_edge_index)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate on validation set
        auc, ap = evaluate_link_prediction(args, model, data)
        if auc > best_auc:
            best_auc = auc
            best_ap = ap

        # Store metrics
        auc_list.append(auc)
        ap_list.append(ap)
        loss_list.append(loss.item())

        # Log metrics to WandB
        # wandb.log({
        #     'epoch': epoch,
        #     'loss': loss.item(),
        #     'AUC': auc,
        #     'AP': ap
        # })

        # Get curvatures and log them
        curvatures = model.get_curvatures()
        curvature_logs.append(curvatures)
        # for key, value in curvatures.items():
            # wandb.log({f'curvature/{key}': value, 'epoch': epoch})

        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

    print(f'Best AUC: {best_auc:.4f}, Best AP: {best_ap:.4f}')

    # After training, print filter weights, component weights, and curvatures
    filter_weights = model.get_filter_weights()
    if filter_weights is not None:
        print('\nFilter Weights (epsilon):')
        print(filter_weights)
        # wandb.log({'filter_weights': wandb.Histogram(filter_weights)})

    component_weights = model.get_component_weights()
    if component_weights is not None:
        print('\nComponent Weights (theta):')
        for idx, theta in enumerate(component_weights):
            print(f'Theta {idx}: {theta}')
            # wandb.log({f'component_weights/theta_{idx}': wandb.Histogram(theta)})

    print('\nLearned Curvatures:')
    final_curvatures = model.get_curvatures()
    for key, value in final_curvatures.items():
        print(f'{key}: {value}')
        # wandb.log({f'final_curvature/{key}': value})

    # Report best metrics to WandB
    # wandb.log({'best_auc': best_auc, 'best_ap': best_ap})

    return (best_auc, best_ap)  # Return the best AUC and AP scores

def link_prediction_loss(model, z, pos_edge_index, neg_edge_index):
    """
    Compute link prediction loss for both positive and negative edges using the inner product decoder.
    """
    # Positive edge loss
    pos_logits = model.decode(z, pos_edge_index)
    pos_labels = torch.ones(pos_logits.size(0), device=pos_logits.device)

    # Negative edge loss
    neg_logits = model.decode(z, neg_edge_index)
    neg_labels = torch.zeros(neg_logits.size(0), device=neg_logits.device)

    # Concatenate positive and negative logits and labels
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def sample_neg_edges_from_mask(neg_adj_mask, num_neg_edges):
    """
    Samples negative edges from the negative adjacency mask.

    Args:
        neg_adj_mask (Tensor): Negative adjacency mask of shape [num_nodes, num_nodes].
        num_neg_edges (int): Number of negative edges to sample.

    Returns:
        Tensor: Negative edge indices of shape [2, num_neg_edges].
    """
    # Get all possible negative edge indices
    neg_edge_indices = torch.nonzero(neg_adj_mask, as_tuple=False).t()  # Shape: [2, num_neg_edges_available]

    num_neg_available = neg_edge_indices.size(1)
    if num_neg_available < num_neg_edges:
        raise ValueError(f"Not enough negative edges to sample: requested {num_neg_edges}, available {num_neg_available}")

    # Randomly permute and select the required number of negative edges
    perm = torch.randperm(num_neg_available)
    neg_edge_index = neg_edge_indices[:, perm[:num_neg_edges]]

    return neg_edge_index

def evaluate_link_prediction(args, model, data):
    """
    Evaluation function for Link Prediction, handling both CUSP and baseline models.
    """
    model.eval()
    with torch.no_grad():
        if args.model == 'cusp':
            z = model.encode(data.x, data.train_pos_edge_index, kappa=data.kappa)
        else:
            # Baseline models (GCN, GAT, SAGE) don't use kappa
            z = model.encode(data.x, data.train_pos_edge_index)

        auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    return auc, ap

if __name__ == '__main__':
    main()