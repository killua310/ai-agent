import networkx as nx
from typing import List, Tuple, Dict, Any
import json
import os

class LocalMemoryStore:
    def __init__(self, storage_file="memory_graph.json"):
        self.storage_file = storage_file
        self.graph = nx.MultiDiGraph() # Support multiple edges between same nodes
        self.load_graph()

    def load_graph(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                print(f"[Memory] Loaded graph from {self.storage_file}")
            except Exception as e:
                print(f"[Memory] Error loading graph: {e}")

    def save_graph(self):
        try:
            data = nx.node_link_data(self.graph)
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Memory] Saved graph to {self.storage_file}")
        except Exception as e:
            print(f"[Memory] Error saving graph: {e}")

    def _find_node(self, entity: str):
        """Helper to find an existing node case-insensitively and handling articles"""
        if entity in self.graph:
            return entity
        
        entity_lower = entity.lower()
        
        for node in self.graph.nodes():
            node_str = str(node).lower()
            if node_str == entity_lower:
                return node
            
            # Handle "the kitchen" == "kitchen"
            if node_str == f"the {entity_lower}" or entity_lower == f"the {node_str}":
                return node
            
            # Handle plurals (simple 's')
            if node_str == f"{entity_lower}s" or entity_lower == f"{node_str}s":
                return node
                
        return None

    def get_primary_user(self):
        """Returns the ID of the first 'person' node added to the graph."""
        # Python 3.7+ preserves insertion order for dicts, so graph.nodes() order 
        # roughly reflects creation order if loaded from JSON linearly.
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "person":
                return node
        return None

    def add_relation(self, subject: str, predicate: str, object_: str, metadata: Dict[str, Any] = None):
        """Adds a relationship to the graph: (Subject) -> [Predicate] -> (Object)"""
        if metadata is None:
            metadata = {}
            
        # Resolve to existing nodes if possible to avoid duplicates like "Mom" vs "mom"
        existing_subject = self._find_node(subject)
        existing_object = self._find_node(object_)
        
        # Use existing ID if found, otherwise use the new one
        s_node = existing_subject if existing_subject else subject
        o_node = existing_object if existing_object else object_
        
        # LOGIC FOR MOVING OBJECTS:
        # If predicate is location-based ('in', 'on', 'at', 'located in'), remove old location
        location_predicates = {'in', 'at', 'on', 'located in', 'is in', 'put ... in'} 
        
        # Simple heuristic: If predicate is strictly "in" or "is in", remove other "in" edges
        if predicate.lower() in ['in', 'is in', 'on', 'at']:
             # Find existing edges from s_node with these predicates and remove them
             if self.graph.has_node(s_node):
                 # Iterate over copy because we might modify
                 for neighbor in list(self.graph.successors(s_node)):
                     edge_data = self.graph.get_edge_data(s_node, neighbor)
                     if edge_data:
                         keys_to_remove = []
                         for key, attr in edge_data.items():
                             if attr.get('relation', '').lower() in ['in', 'is in', 'on', 'at']:
                                 keys_to_remove.append(key)
                         
                         for k in keys_to_remove:
                             self.graph.remove_edge(s_node, neighbor, key=k)
                             print(f"[Memory] Removed stale location: {s_node} -[in]-> {neighbor}")

        # Add nodes (if they don't exist)
        self.graph.add_node(s_node)
        self.graph.add_node(o_node)
        
        # Prepare edge attributes
        edge_attrs = {'relation': predicate}
        edge_attrs.update(metadata) # Merge metadata (timestamp, source, etc.)
        
        # Deduplication: Check if this exact relation exists
        if self.graph.has_edge(s_node, o_node):
             # iterate keys
             edge_data = self.graph.get_edge_data(s_node, o_node)
             for key, attr in edge_data.items():
                 existing_rel = attr.get('relation', '')
                 # Normalized comparison (ignore case and whitespace)
                 if existing_rel.lower().strip() == predicate.lower().strip():
                     # Update metadata only
                     for k, v in metadata.items():
                         self.graph[s_node][o_node][key][k] = v
                     print(f"[Memory] Updated existing: {s_node} -[{predicate}]-> {o_node} | Meta updated")
                     self.save_graph()
                     return

        # Add edge with the predicate and metadata
        self.graph.add_edge(s_node, o_node, **edge_attrs)
        print(f"[Memory] Stored: {s_node} -[{predicate}]-> {o_node} | Meta: {metadata}")
        
        # Auto-save on every update
        self.save_graph()

    def add_node_type(self, entity: str, type_: str):
        """Stores the type of the entity (e.g., person, organization)"""
        node = self._find_node(entity)
        if not node:
            node = entity
            self.graph.add_node(node)
            
        # Only set if not already set or if updating from 'concept'/'object' to 'person' (upgrade)
        current_type = self.graph.nodes[node].get('type', 'unknown')
        if current_type == 'unknown' or type_ in ['person', 'organization']:
             self.graph.nodes[node]['type'] = type_
             print(f"[Memory] Set Type: {node} -> {type_}")
             self.save_graph()

    def remove_node(self, entity: str):
        """Soft Deletes a node: Removes attributes and OUTGOING edges, but keeps the node so incoming links remain."""
        node = self._find_node(entity)
        if node:
            # 1. Clear Attributes (reset to empty)
            self.graph.nodes[node].clear()
            
            # 2. Remove Outgoing Edges (What this entity knows)
            # list() is needed to avoid iterator modification issues
            for neighbor in list(self.graph.successors(node)):
                self.graph.remove_edge(node, neighbor)
                
            print(f"[Memory] Soft deleted node: {node} (Preserved incoming links)")
            self.save_graph()
            return True
        return False

    def add_attribute(self, entity: str, attribute: str, value: str):
        """Adds an attribute to a node (not a separate node). e.g. Minji -> is -> pretty"""
        node = self._find_node(entity)
        if not node:
            node = entity
            self.graph.add_node(node)
        
        # Initialize attributes dict if not present
        if 'attributes' not in self.graph.nodes[node]:
            self.graph.nodes[node]['attributes'] = {}
            
        current_val = self.graph.nodes[node]['attributes'].get(attribute)
        
        if current_val is None:
            # New attribute
            self.graph.nodes[node]['attributes'][attribute] = value
        elif isinstance(current_val, list):
            # Already a list, append if unique
            if value not in current_val:
                current_val.append(value)
        else:
            # Convert existing single value to list if different
            if current_val != value:
                self.graph.nodes[node]['attributes'][attribute] = [current_val, value]
                
        print(f"[Memory] Added Attribute: {node}.{attribute} = {value}")
        self.save_graph()

    def search_memory(self, query: str) -> List[str]:
        """Searches for a query string across all nodes, attributes, and relations."""
        query_lower = query.lower()
        results = []
        
        # 1. Search Nodes & Attributes
        for node, data in self.graph.nodes(data=True):
            node_str = str(node)
            matches = False
            
            # Check Node ID
            if query_lower in node_str.lower():
                matches = True
                
            # Check Attributes
            attributes = data.get('attributes', {})
            for attr, value in attributes.items():
                if query_lower in str(value).lower() or query_lower in str(attr).lower():
                    matches = True
                    # Add specific attribute match
                    results.append(f"{node} ({attr}: {value})")
            
            # Use 'type' if present
            node_type = data.get('type', '')
            if query_lower in str(node_type).lower():
                matches = True
                
            if matches:
                # Add general node existence
                # Get all attributes for context
                attr_str = ", ".join([f"{k}={v}" for k, v in attributes.items()])
                results.append(f"Node: {node} [{node_type}] {{ {attr_str} }}")

        # 2. Search Relations (Edges)
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            if query_lower in str(relation).lower() or query_lower in str(v).lower() or query_lower in str(u).lower():
                results.append(f"{u} {relation} {v}")
                
        # Deduplicate
        return list(set(results))

    def query_relations(self, entity: str) -> List[Tuple[str, str, str, Dict]]:
        """Finds all relationships connected to a specific entity"""
        results = []
        
        # Find the node (Case-insensitive match)
        target_node = None
        if entity in self.graph:
            target_node = entity
        else:
            for node in self.graph.nodes():
                if str(node).lower() == entity.lower():
                    target_node = node
                    break
        
        if target_node:
            # 1. Get Node Attributes
            attrs = self.graph.nodes[target_node].get('attributes', {})
            for key, val in attrs.items():
                results.append((target_node, key, val, {})) # No meta for attributes yet

            # 2. Get Graph Relations (BFS)
            visited = set()
            queue = [(target_node, 0)]
            visited.add(target_node)
            
            while queue:
                current, depth = queue.pop(0)
                if depth >= 2:
                    continue
                
                # Successors
                for neighbor in self.graph.successors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                    
                    # Handle multiple edges (MultiDiGraph)
                    edge_data = self.graph.get_edge_data(current, neighbor)
                    # edge_data is a dict of keys -> attributes {0: {'relation': '...'}, 1: ...}
                    for key in edge_data:
                        attrs = edge_data.get(key, {})
                        relation = attrs.get('relation', 'related_to')
                        results.append((current, relation, neighbor, attrs))
                
                # Predecessors
                for predecessor in self.graph.predecessors(current):
                    if predecessor not in visited:
                        visited.add(predecessor)
                        queue.append((predecessor, depth + 1))
                        
                    # Handle multiple edges
                    edge_data = self.graph.get_edge_data(predecessor, current)
                    for key in edge_data:
                        attrs = edge_data.get(key, {})
                        relation = attrs.get('relation', 'related_to')
                        results.append((predecessor, relation, current, attrs))
        return results

    def get_all_triplets(self) -> List[str]:
        """Returns all knowledge as text for context"""
        triplets = []
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            triplets.append(f"{u} {relation} {v}")
        return triplets

    def get_graph_data(self) -> Dict[str, Any]:
        """Returns nodes and links for frontend visualization"""
        return nx.node_link_data(self.graph)
