import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

# Initialize colorama for Windows
init()

def build_graph(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['sender'], row['receiver'], 
                  amount=row['amount'], 
                  timestamp=row['timestamp'], 
                  is_fraud=row['is_fraud'],
                  device_id=row['device_id'])
    fraud_nodes = set(df[df['is_fraud'] == 1]['sender']).union(set(df[df['is_fraud'] == 1]['receiver']))
    for node in G.nodes():
        G.nodes[node]['is_fraud'] = 1 if node in fraud_nodes else 0
    return G

def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    colors = ['red' if G.nodes[node]['is_fraud'] == 1 else 'green' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    edge_labels = {(u, v): f"${d['amount']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("Fraud Detection Graph\n(Red: Fraud, Green: Normal)", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def detect_patterns(G):
    print(f"{Fore.CYAN}=== Pattern Detection Results ==={Style.RESET_ALL}")
    
    cycles = list(nx.simple_cycles(G))
    print(f"{Fore.YELLOW}Fraud Rings (Cycles):{Style.RESET_ALL}", 
          f"{Fore.RED}{cycles}{Style.RESET_ALL}" if cycles else "None detected")
    
    centrality = nx.degree_centrality(G)
    hubs = {node: score for node, score in centrality.items() if score > 0.2}
    print(f"{Fore.YELLOW}Highly Connected Hubs:{Style.RESET_ALL}", hubs)
    
    components = list(nx.weakly_connected_components(G))
    print(f"{Fore.YELLOW}Isolated Components:{Style.RESET_ALL}", [comp for comp in components if len(comp) > 1])
    
    node_freq = {node: G.out_degree(node) for node in G.nodes()}
    frequent_nodes = {node: freq for node, freq in node_freq.items() if freq > 3}
    print(f"{Fore.YELLOW}High Transaction Frequency Nodes:{Style.RESET_ALL}", frequent_nodes)
    
    device_to_edges = {}
    for u, v, data in G.edges(data=True):
        device = data['device_id']
        if device not in device_to_edges:
            device_to_edges[device] = []
        device_to_edges[device].append((u, v))
    suspicious_devices = {device: edges for device, edges in device_to_edges.items() 
                         if len(edges) > 3 or any(G.nodes[u]['is_fraud'] or G.nodes[v]['is_fraud'] for u, v in edges)}
    print(f"{Fore.YELLOW}Suspicious Device Patterns:{Style.RESET_ALL}", suspicious_devices)

if __name__ == "__main__":
    G = build_graph("data/transactions.csv")
    visualize_graph(G)
    detect_patterns(G)