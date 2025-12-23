from app.memory import LocalMemoryStore

def test_query():
    store = LocalMemoryStore()
    print("Graph Nodes:", store.graph.nodes(data=True))
    
    entity = "Minji"
    print(f"\nQuerying relations for '{entity}'...")
    relations = store.query_relations(entity)
    print("Relations found:", relations)

if __name__ == "__main__":
    test_query()
