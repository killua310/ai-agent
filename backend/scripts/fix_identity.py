
import json
import networkx as nx
from networkx.readwrite import json_graph

# Load Graph
path = "backend/memory_graph.json"
with open(path, 'r') as f:
    data = json.load(f)

G = json_graph.node_link_graph(data)

# Fix Identity
if G.has_node("I"):
    print("Found 'I' node. Merging into 'Miccel'...")
    
    # Move edges
    edges_to_add = []
    
    # Outgoing from I
    for _, target, data in G.edges("I", data=True):
        edges_to_add.append(("Miccel", target, data))
        
    # Incoming to I
    for source, _, data in G.in_edges("I", data=True):
        edges_to_add.append((source, "Miccel", data))
        
    # Move attributes
    i_attrs = G.nodes["I"].get("attributes", {})
    miccel_attrs = G.nodes["Miccel"].get("attributes", {}) if G.has_node("Miccel") else {}
    
    # Merge attrs (Miccel wins conflicts, but take new ones from I)
    for k, v in i_attrs.items():
        if k not in miccel_attrs:
            miccel_attrs[k] = v
            
    # Apply changes
    G.add_edge_from = edges_to_add # Wait, networkx syntax error. G.add_edges_from
    G.add_edges_from(edges_to_add)
    
    if G.has_node("Miccel"):
        G.nodes["Miccel"]["attributes"] = miccel_attrs
    else:
        # If Miccel didn't exist (unlikely), create it with I's attrs
        G.add_node("Miccel", type="person", attributes=miccel_attrs)

    G.remove_node("I")
    print("Done. 'I' merged into 'Miccel'.")
    
    # Save
    with open(path, 'w') as f:
        json.dump(json_graph.node_link_data(G), f, indent=2)
else:
    print("'I' node not found.")
