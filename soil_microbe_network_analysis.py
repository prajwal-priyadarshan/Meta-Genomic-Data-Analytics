# ============================================================
# 🧬 Soil Microbial Co-occurrence Network Analysis
# ============================================================

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# ============================================================
# STEP 1 — Load & Clean Data
# ============================================================

# Change this to your dataset filename
filename = r"Dataset\Soil_microbe_dataset.csv"
df = pd.read_csv(filename)

# Convert range strings (like "10–20") to mean values
def convert_range(val):
    if isinstance(val, str):
        if "–" in val:  # en dash (not hyphen)
            parts = val.split("–")
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return None
        try:
            return float(val)
        except:
            return None
    return val

# Apply cleaning
df = df.applymap(convert_range)

# Drop all-non-numeric columns & missing values
df = df.dropna(axis=1, how="all")
df = df.dropna(axis=0, how="any")

print(f"✅ Cleaned dataset shape: {df.shape}")
print("✅ Numeric columns retained:", list(df.columns))

# ============================================================
# STEP 2 — Correlation Matrix
# ============================================================

corr = df.corr(method="spearman")
print("✅ Correlation matrix computed.")

# ============================================================
# STEP 3 — Build Network from Correlations
# ============================================================

threshold = 0.6  # Only include strong correlations
G = nx.Graph()

# Add nodes
for col in corr.columns:
    G.add_node(col)

# Add edges where |r| >= threshold
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        weight = corr.iloc[i, j]
        if abs(weight) >= threshold:
            G.add_edge(corr.columns[i], corr.columns[j], weight=weight)

print(f"✅ Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# ============================================================
# STEP 4 — Community Detection (Clustering)
# ============================================================

communities = list(greedy_modularity_communities(G))
community_map = {}
for cid, comm in enumerate(communities):
    for node in comm:
        community_map[node] = cid

print(f"✅ Detected {len(communities)} communities in the network.")

# ============================================================
# STEP 5 — Network Visualization
# ============================================================

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.6, seed=42)

# Node color by community & size by degree
node_colors = [community_map[n] for n in G.nodes()]
node_sizes = [500 + 200 * G.degree(n) for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       cmap=plt.cm.Set3, alpha=0.9)

# Edge color by correlation strength
edges = nx.draw_networkx_edges(
    G, pos,
    edge_color=[d['weight'] for _, _, d in G.edges(data=True)],
    edge_cmap=plt.cm.coolwarm, width=2, alpha=0.5
)

nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

plt.colorbar(edges, label="Spearman Correlation")
plt.title("🧬 Soil Microbial Co-occurrence Network (Clustered)", fontsize=15)
plt.axis("off")
plt.show()

# ============================================================
# STEP 6 — Network Statistics & Analysis
# ============================================================

print("\n📊 --- Network Statistics ---")
density = nx.density(G)
modularity = nx.community.modularity(G, communities)
clustering_coeff = nx.average_clustering(G)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

print(f"Network Density: {density:.3f}")
print(f"Network Modularity: {modularity:.3f}")
print(f"Average Clustering Coefficient: {clustering_coeff:.3f}")

# Top 10 Hub Nodes (by Degree Centrality)
top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n🌱 Top 10 Hub Microbes (by Degree Centrality):")
for name, val in top_hubs:
    print(f"  {name}: {val:.3f}")

# ============================================================
# STEP 7 — Export Network for Gephi / Cytoscape
# ============================================================

# Create Nodes DataFrame
node_data = pd.DataFrame({
    'Node': list(G.nodes()),
    'Degree': [G.degree(n) for n in G.nodes()],
    'Betweenness': [betweenness_centrality[n] for n in G.nodes()],
    'Community': [community_map[n] for n in G.nodes()]
})

# Create Edges DataFrame
edge_data = pd.DataFrame([
    {'Source': u, 'Target': v, 'Weight': d['weight']}
    for u, v, d in G.edges(data=True)
])

# Save to CSV
node_data.to_csv("network_nodes.csv", index=False)
edge_data.to_csv("network_edges.csv", index=False)

print("\n✅ Exported for Gephi/Cytoscape:")
print("  → network_nodes.csv")
print("  → network_edges.csv")

# ============================================================
# STEP 8 — Optional: Summary Report
# ============================================================

summary = {
    "Nodes": G.number_of_nodes(),
    "Edges": G.number_of_edges(),
    "Communities": len(communities),
    "Density": density,
    "Modularity": modularity,
    "Avg_Clustering": clustering_coeff,
    "Top_Hubs": [n for n, _ in top_hubs]
}

pd.DataFrame([summary]).to_csv("network_summary.csv", index=False)
print("✅ Network summary saved to network_summary.csv")
