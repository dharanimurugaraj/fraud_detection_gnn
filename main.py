import json
from graph_builder import build_graph, visualize_graph, detect_patterns
from gnn_model import prepare_data, train_model, evaluate_model, load_model

def export_graph(G, pred):
    node_risk = {node: "high" if pred[i] > 0.5 else "low" for i, node in enumerate(G.nodes())}
    graph_json = {
        "nodes": [{"id": node, "risk": node_risk[node]} for node in G.nodes()],
        "edges": [{"from": u, "to": v} for u, v in G.edges()]
    }
    with open("graph_output.json", "w") as f:
        json.dump(graph_json, f, indent=2)
    print("Graph exported to graph_output.json")

def main():
    # Build and analyze graph
    G = build_graph("data/transactions.csv")
    visualize_graph(G)
    detect_patterns(G)
    
    # Train GNN and run inference
    data = prepare_data(G)
    try:
        model = load_model(data)  # Try loading existing model
        pred = evaluate_model(model, data)
    except FileNotFoundError:
        print("No saved model found, training a new one...")
        model = train_model(data)  # Train if no model exists
        pred = evaluate_model(model, data)
    
    # Export results
    export_graph(G, pred)

if __name__ == "__main__":
    main()