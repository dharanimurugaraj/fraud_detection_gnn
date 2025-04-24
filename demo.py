import torch
from graph_builder import build_graph, visualize_graph, detect_patterns
from gnn_model import prepare_data, load_model, evaluate_model
from main import export_graph

def run_demo():
    print("=== Fraud Detection Demo ===")
    G = build_graph("data/transactions.csv")
    visualize_graph(G)
    detect_patterns(G)
    
    data = prepare_data(G)
    model = load_model(data)
    pred = evaluate_model(model, data)
    export_graph(G, pred)
    print("Demo complete! Check graph_output.json for results.")

if __name__ == "__main__":
    run_demo()