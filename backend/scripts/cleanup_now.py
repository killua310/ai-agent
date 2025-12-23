
import sys
import os
import json

# Add backend directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

from app.memory import LocalMemoryStore
from app.agent import SecondBrainAgent

def cleanup():
    print("Initializing Memory and Agent...")
    memory = LocalMemoryStore(storage_file=os.path.join(backend_dir, "memory_graph.json"))
    agent = SecondBrainAgent(memory)
    
    # Get all nodes
    all_nodes = list(memory.graph.nodes())
    print(f"Found {len(all_nodes)} nodes. Running consolidation...")
    
    # Create a fake state with all entities to trigger full graph check
    state = {
        "entities": all_nodes,
        "input": "cleanup",
        "chat_history": []
    }
    
    # Run the consolidation functionality directly
    # Note: We need to mock the LLM or ensure API keys are set in environment
    if not agent.llm:
        print("ERROR: No LLM available (Check API Keys).")
        return

    agent.node_consolidate_memory(state)
    print("Cleanup Complete.")

if __name__ == "__main__":
    cleanup()
