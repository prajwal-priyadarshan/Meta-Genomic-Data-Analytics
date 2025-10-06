# ============================================================
# 🌍 Integrative Eco-Metabolomic Map
# ============================================================

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Step 1 — Load Existing Network Files & Function Predictions
# ------------------------------------------------------------
nodes = pd.read_csv(r"network_nodes.csv")     # from your network export
edges = pd.read_csv(r"network_edges.csv")     # from your network export

# Optionally, bring in functional role predictions (from part B)
# Suppose you have a file or DataFrame mapping Node -> PredictedFunction
try:
    func_df = pd.read_csv(r"functional_predictions.csv")  # Node, PredictedFunction
except:
    # demo mapping (random for visualization)
    np.random.seed(42)
    func_df = pd.DataFrame({
        "Node": nodes["Node"],
        "PredictedFunction": np.random.choice(
            ["CarbonFixation", "NitrogenCycle", "Decomposer"], size=len(nodes)
        )
    })

# ------------------------------------------------------------
# Step 2 — Build Graph Object
# ------------------------------------------------------------
G = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='Weight')

# Merge node-level attributes
for _, row in nodes.iterrows():
    if row['Node'] in G:
        G.nodes[row['Node']]['Degree'] = row['Degree']
        G.nodes[row['Node']]['Community'] = row['Community']

# Merge functional predictions
# func_map = dict(zip(func_df['Node'], func_df['PredictedFunction']))

# File: eco_metabolic_map.py (around line 42)
# Correction: Use 'ID' instead of 'Node' and 'Predicted_Functional_Role' instead of 'PredictedFunction'

func_map = dict(zip(func_df['ID'], func_df['Predicted_Functional_Role']))
nx.set_node_attributes(G, func_map, 'Function')



# ------------------------------------------------------------
# Step 3 — Visualization as Eco-Metabolomic Map
# ------------------------------------------------------------
plt.figure(figsize=(13, 11))
pos = nx.spring_layout(G, k=0.6, seed=42)

# Define colors by function type
func_colors = {
    "CarbonFixation": "#2ca02c",   # green
    "NitrogenCycle": "#1f77b4",    # blue
    "Decomposer": "#ff7f0e",       # orange
    "SulfurReduction": "#9467bd"   # purple
}
node_colors = [func_colors.get(G.nodes[n].get('Function', ''), "#d3d3d3") for n in G.nodes()]
node_sizes = [400 + 300 * G.degree(n) for n in G.nodes()]

# Draw nodes + edges
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
edges = nx.draw_networkx_edges(
    G, pos,
    edge_color=[d['Weight'] for _, _, d in G.edges(data=True)],
    edge_cmap=plt.cm.coolwarm, width=2, alpha=0.5
)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

plt.title("🌱 Eco-Metabolomic Map of Soil Microbial Network", fontsize=16)
plt.colorbar(edges, label="Spearman Correlation (Interaction Strength)")
plt.axis("off")

# Custom legend
for name, color in func_colors.items():
    plt.scatter([], [], c=color, label=name, s=200)
plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Predicted Functional Roles",loc='lower left')
plt.show()
