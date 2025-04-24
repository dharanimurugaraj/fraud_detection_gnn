import streamlit as st
import pandas as pd
import networkx as nx
import json
import torch
import os

from graph_builder import build_graph, detect_patterns
from gnn_model import prepare_data, train_model, evaluate_model, load_model

# Streamlit page configuration
st.set_page_config(page_title="Fraud Detection GNN", layout="wide")

# Sidebar
st.sidebar.title("Fraud Detection System")
st.sidebar.markdown("Upload a transaction dataset and analyze fraud using a Graph Neural Network (GNN).")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Transactions CSV", type=["csv"])
start_analysis = st.sidebar.button("Start Analysis")

# Main logic
if start_analysis and uploaded_file is not None:
    # Save uploaded file
    data_path = "data/transactions.csv"
    os.makedirs("data", exist_ok=True)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.title("Fraud Detection Dashboard")

    # Build graph
    G = build_graph(data_path)

    # Interactive network graph
    st.subheader("Interactive Transaction Graph")
    from pyvis.network import Network

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    for node in G.nodes():
        color = "red" if G.nodes[node]['is_fraud'] == 1 else "green"
        net.add_node(node, label=str(node), color=color)
    for u, v in G.edges():
        net.add_edge(u, v)
    net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 16,
              "strokeWidth": 2
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true
              }
            }
          },
          "physics": {
            "enabled": true
          }
        }
    """)
    net.save_graph("network.html")
    st.components.v1.html(open("network.html", "r", encoding="utf-8").read(), height=600, scrolling=True)

    # Pattern detection
    st.subheader("Detected Fraud Patterns")
    with st.expander("Click to view detected patterns"):
        detect_patterns(G)

    # GNN model inference
    st.subheader("GNN Model Inference")
    data = prepare_data(G)

    try:
        model = load_model(data)
        st.success("Pre-trained model loaded successfully.")
    except FileNotFoundError:
        st.warning("No pre-trained model found. Training a new model...")
        model = train_model(data)
        st.success("Model trained successfully.")

    pred = evaluate_model(model, data)

    # Display risk levels
    node_risk = {node: "High Risk" if pred[i] > 0.5 else "Low Risk" for i, node in enumerate(G.nodes())}
    pred_df = pd.DataFrame([{"Node": node, "Risk": risk} for node, risk in node_risk.items()])
    st.subheader("Predicted Risk per Node")
    st.dataframe(pred_df)

    # Export to JSON
    graph_json = {
        "nodes": [{"id": node, "risk": node_risk[node]} for node in G.nodes()],
        "edges": [{"from": u, "to": v} for u, v in G.edges()]
    }

    with open("graph_output.json", "w") as f:
        json.dump(graph_json, f, indent=2)

    st.success("Graph exported with risk scores to `graph_output.json`")

elif start_analysis and uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
