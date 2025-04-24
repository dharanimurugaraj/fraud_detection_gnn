import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from colorama import init, Fore, Style

init()

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

def prepare_data(G):
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = G.number_of_nodes()
    x = torch.eye(num_nodes, dtype=torch.float)
    y = torch.tensor([G.nodes[node]['is_fraud'] for node in G.nodes()], dtype=torch.float)
    train_mask = torch.rand(num_nodes) < 0.8
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

def train_model(data, epochs=100, save_path="fraud_model.pth"):
    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=16, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(out[data.train_mask], data.y[data.train_mask].unsqueeze(1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = (out > 0.5).float().squeeze()
        y_true = data.y.numpy()
        y_pred = pred.numpy()
        acc = accuracy_score(y_true[data.train_mask], y_pred[data.train_mask])
        prec = precision_score(y_true[data.train_mask], y_pred[data.train_mask])
        rec = recall_score(y_true[data.train_mask], y_pred[data.train_mask])
        print(f"{Fore.GREEN}=== Model Evaluation ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Accuracy:{Style.RESET_ALL} {acc:.4f}")
        print(f"{Fore.YELLOW}Precision:{Style.RESET_ALL} {prec:.4f}")
        print(f"{Fore.YELLOW}Recall:{Style.RESET_ALL} {rec:.4f}")
    return pred

def load_model(data, model_path="fraud_model.pth"):
    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=16, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

if __name__ == "__main__":
    from graph_builder import build_graph
    G = build_graph("data/transactions.csv")
    data = prepare_data(G)
    model = train_model(data)
    evaluate_model(model, data)