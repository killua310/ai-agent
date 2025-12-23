from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.memory import LocalMemoryStore
import json
import os
import datetime

# Define the Rational Agent State
class AgentState(TypedDict):
    input: str                  # Original user query
    chat_history: List[str]     # Conversation context
    
    # -- PREPROCESSING --
    rewritten_input: str        # After coref correction (e.g. "He" -> "Miccel")
    entities: List[str]         # Canonical entity names mentioned
    intent: Literal["store", "query", "chat"] 

    # -- MEMORY & RETRIEVAL --
    extracted_triplets: List[List[str]] # For storage path
    context: List[str]          # Retrieved facts for query path
    
    # -- REASONING --
    plan: List[str]             # Sub-queries or steps
    context: List[str]          # Retrieved facts
    visualize_targets: List[str] # Entities to visualize
    reasoning_trace: Dict[str, Any] # {direct_facts: [], inferred_facts: []}
    
    # -- OUTPUT --
    response: str

# Define Structured Output for Extraction
class Triplet(BaseModel):
    subject: str = Field(description="The subject of the relationship or attribute")
    subject_type: Literal["person", "organization", "location", "object", "concept"] = Field(description="Type of the subject entity")
    predicate: str = Field(description="The relationship type or attribute name")
    object_: str = Field(description="The object of the relationship or attribute value")
    object_type: Literal["person", "organization", "location", "object", "concept", "value"] = Field(description="Type of the object entity (use 'value' for attributes)")
    type: Literal["relation", "attribute"] = Field(description="Type of information: 'relation' for node-to-node connection, 'attribute' for node property")

class ExtractionResult(BaseModel):
    triplets: List[Triplet] = Field(description="List of extracted knowledge triplets")

class SecondBrainAgent:
    def __init__(self, memory_store: LocalMemoryStore):
        self.memory = memory_store
        
        # Initialize LLM
        openai_key = os.getenv("OPENAI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if openai_key:
            print("Using OpenAI GPT Model (gpt-4o-mini)")
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        elif google_key:
            print("Using Google Gemini Model")
            self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        else:
             print("CRITICAL WARNING: No API Key found. Agent will fail.")
             self.llm = None

        # Build the Rational Pipeline Graph
        self.workflow = StateGraph(AgentState)
        
        # -- ADD NODES --
        self.workflow.add_node("resolve_coref", self.node_resolve_coref)
        self.workflow.add_node("entity_linker", self.node_entity_linker)
        self.workflow.add_node("classify_intent", self.node_classify_intent)
        
        # Paths
        self.workflow.add_node("store_memory", self.node_store_memory)
        self.workflow.add_node("generate_storage_response", self.node_generate_storage_response)
        self.workflow.add_node("general_chat", self.node_general_chat)
        self.workflow.add_node("visualize_graph", self.node_visualize_graph)
        self.workflow.add_node("consolidate_memory", self.node_consolidate_memory)
        self.workflow.add_node("check_ambiguity", self.node_check_ambiguity)
        
        # Query Pipeline
        self.workflow.add_node("query_planner", self.node_query_planner)
        self.workflow.add_node("retrieval_node", self.node_retrieval)
        self.workflow.add_node("reasoning_node", self.node_reasoning)
        self.workflow.add_node("answer_node", self.node_answer) # Assuming this is the answer node logic
        self.workflow.add_node("delete_memory", self.node_delete_memory)

        # -- ADD EDGES --
        self.workflow.set_entry_point("resolve_coref")
        self.workflow.add_edge("resolve_coref", "entity_linker")
        self.workflow.add_edge("entity_linker", "classify_intent")
        
        def route_intent(state: AgentState):
            intent = state.get("intent", "chat")
            if intent == "store":
                return "store_memory"
            elif intent == "delete":
                return "delete_memory"
            elif intent == "confirmed_delete":
                return "delete_memory"
            elif intent == "query":
                return "query_planner" 
            elif intent == "visualize":
                return "visualize_graph"
            else:
                return "general_chat"

        self.workflow.add_conditional_edges(
            "classify_intent",
            route_intent,
            {
                "store_memory": "check_ambiguity", # Redirect store to check ambiguity first
                "delete_memory": "delete_memory",
                "query_planner": "query_planner",
                "visualize_graph": "visualize_graph",
                "general_chat": "general_chat"
            }
        )
        
        # Ambiguity Check Routing
        def route_ambiguity(state: AgentState):
            # If the ambiguity node produced a response (question), stop and return it.
            if state.get("response"):
                return END
            return "store_memory"

        self.workflow.add_conditional_edges(
            "check_ambiguity",
            route_ambiguity,
            {
                END: END,
                "store_memory": "store_memory"
            }
        )
        
        # Connect Main Paths
        # Store -> Generate Response -> Consolidate -> End
        self.workflow.add_edge("store_memory", "generate_storage_response")
        self.workflow.add_edge("generate_storage_response", "consolidate_memory")
        self.workflow.add_edge("consolidate_memory", END)

        self.workflow.add_edge("delete_memory", END)
        self.workflow.add_edge("general_chat", END)
        self.workflow.add_edge("visualize_graph", END)
        
        # Connect Query Path
        self.workflow.add_edge("query_planner", "retrieval_node")
        self.workflow.add_edge("retrieval_node", "reasoning_node")
        self.workflow.add_edge("reasoning_node", "answer_node")
        self.workflow.add_edge("answer_node", END)


        self.app = self.workflow.compile()

    # --- PREPROCESSING NODES ---
    
    def node_resolve_coref(self, state: AgentState):
        """Step 1: Rewrite input to resolve pronouns using history."""
        # Check if history exists
        history_text = "\n".join(state.get("chat_history", [])[-5:]) # Last 5 turns
        
        if not history_text:
            return {"rewritten_input": state["input"]}

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Coreference Resolution system. Rewrite the user's latest query to resolve ALL pronouns (he, she, it, they) to specific names found in the Chat History.\n"
                       "- USER IDENTITY: The user's name is 'Miccel'. ALWAYS resolve 'I', 'me', 'my' to 'Miccel'.\n"
                       "- Input: 'Where does he work?' (History: 'Tell me about Miccel') -> Output: 'Where does Miccel work?'\n"
                       "- Input: 'Is it cold there?' (History: 'I am in Magok') -> Output: 'Is it cold in Magok?'\n"
                       "- ANSWERING CLARIFICATION: If the history shows the agent asked a question (e.g., 'Brother or friend?'), combine the user's answer with the original context.\n"
                       "  - History: 'User: Beomgyu is my bro. Agent: Brother or friend?' Input: 'Friend' -> Output: 'Beomgyu is my friend.'\n"
                       "- If no pronouns or no history context, return the input exactly as is.\n"
                       "- RETURN ONLY THE REWRITTEN SENTENCE."),
            ("user", "Chat History:\n{history}\n\nCurrent Input: {input}")
        ])
        chain = prompt | self.llm
        rewritten = chain.invoke({"input": state["input"], "history": history_text}).content.strip()
        print(f"[Coreference] Original: {state['input']} -> Rewritten: {rewritten}")
        return {"rewritten_input": rewritten}

    def node_entity_linker(self, state: AgentState):
        """Step 2: Identify and link entities from the rewritten input."""
        # Simple extraction for now, can be enhanced with fuzzy matching against graph nodes later
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Identify all key entities in the text. Return them as a COMMA-SEPARATED list.\n"
                       "- Entities: People, Organizations, Locations, Objects.\n"
                       "- Simplify: 'Miccel's mom' -> 'Miccel', 'Mom'.\n"
                       "- Examples: 'Miccel works at LG' -> 'Miccel, LG'."),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        entities_str = chain.invoke({"input": state["rewritten_input"]}).content.strip()
        
        entities = [e.strip().replace("'", "").replace('"', "") for e in entities_str.split(',') if e.strip()]
        
        # Canonicalization (Basic) - e.g. "lg" -> "LG"
        # In a real system, we would query the vector store/graph for closest match.
        # For now, we trust the LLM or pass through.
        
        return {"entities": entities}

    def node_classify_intent(self, state: AgentState):
        # 1. Check for Confirmation Context (Stateless Check via History)
        history = state.get("chat_history", [])
        if history:
            # Check last 2 messages in case current input was appended
            last_msgs = history[-2:]
            
            print(f"[Classifier] Checking history for confirmation. Last msgs: {last_msgs}")

            # Look for the specific confirmation string in RECENT history
            confirm_request_found = False
            target_phrase = "Are you sure you want to permanently delete"
            
            for msg in reversed(last_msgs):
                # Handle LangChain Message Objects
                if hasattr(msg, 'content'):
                    content = msg.content
                # Handle Strings (e.g. "AI: Are you sure...")
                elif isinstance(msg, str):
                    content = msg
                else:
                    content = str(msg)
                
                if target_phrase in content:
                     confirm_request_found = True
                     break
            
            if confirm_request_found:
                 # Check if user says Yes (Relaxed check)
                 cleaned_input = state["rewritten_input"].lower().strip(".! ")
                 if cleaned_input in ["yes", "yes please", "confirm", "do it", "sure", "y", "yeah", "yep"]:
                     print(f"[Classifier] Detected Confirmation for Deletion")
                     return {"intent": "confirmed_delete"}
                 else:
                     print(f"[Classifier] Deletion cancelled (Input was '{cleaned_input}')")
                     return {"intent": "chat"} # Treat as cancellation/chat

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Semantic Intent Classifier. Classify the user's input into one of categories:\n"
                       "- 'store': The user is providing new facts, memories, or updates to be saved.\n"
                       "- 'query': The user is asking a question that involves specific ENTITIES (e.g. people, places, organizations) or personal memories. \n"
                       "    - Even if it looks like a general question (e.g. 'Is Korea University good?'), if it involves a specific named entity, classify as 'query' so we can check for personal connections.\n"
                       "- 'delete': The user explicitly wants to DELETE, REMOVE, or FORGET a specific entity or memory.\n"
                       "    - Example: 'Delete Minseok', 'Forget about LG', 'Remove the node for Korea University'.\n"
                       "- 'chat': The user is engaging in casual conversation, offering greetings, or asking purely abstract questions with NO specific entities.\n"
                       "    - Example: 'Hi there', 'How are you?', 'What is love?'.\n\n"
                       "Return ONLY one word: 'store', 'query', 'delete', or 'chat'."),
            ("user", "Input: {input}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"input": state["rewritten_input"]})
        intent = response.content.strip().lower()
        
        if intent not in ["store", "query", "delete", "chat"]:
            intent = "chat"
            
        print(f"[Classifier] Intent: {intent}")
        return {"intent": intent}

    def node_check_ambiguity(self, state: AgentState):
        """Step (Ambiguity Check): Ask clarifying questions for ambiguous relationships."""
        input_text = state["rewritten_input"]
        
        # Fast check for keywords that are often ambiguous
        # We can also use LLM for this, but a keyword check + LLM confirm is robust.
        # Let's use strict LLM to be safe and context-aware.
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Data Accuracy Guardian. Your job is to prevent ambiguous data from entering the Knowledge Graph.\n"
                       "Analyze the input for specific ambiguous terms:\n"
                       "- 'bro' -> Could be 'brother' (kinship) or 'close friend'.\n"
                       "- 'partner' -> Could be 'spouse' or 'business partner'.\n"
                       "- 'connected with' -> Too vague.\n"
                       "- 'it', 'he', 'she' -> If the Coreference Resolution step failed to replace these with names, it's ambiguous.\n\n"
                       "TASK:\n"
                       "1. If the input contains such ambiguity that requires clarification, generate a polite question to ask the user.\n"
                       "   - Example: Input 'He is my bro', Output: 'Just to clarify, do you mean he is your biological brother or a close friend?'\n"
                       "2. If the input is clear (e.g., 'He is my brother', 'They are dating'), return 'CLEAR'.\n"
                       "3. If it's just a general chat/greeting, return 'CLEAR'.\n\n"
                       "RETURN ONLY the question or the word 'CLEAR'.v"),
            ("user", "Input: {input}")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({"input": input_text}).content.strip()
        
        if result.upper() != "CLEAR":
            return {"response": result} # Return question, stops flow
            
        return {} # Continue to store_memory

    def node_store_memory(self, state: AgentState):
        """Extracts entities and stores them in the graph using Rewritten Input."""
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        
        # Use rewritten input for better extraction
        input_text = state["rewritten_input"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "INPUTS: (1) recent chat history, (2) current input.\n"
             "OUTPUT: triplets matching the JSON schema. Follow these rules:\n"
             "1) PRONOUN & LOCATION RESOLUTION\n"
             "   - Use the last turn(s) of history to resolve 'I, me, my, she, he, they, it, there, here'.\n"
             "   - Replace pronouns with the concrete entity name.\n\n"
             "2) NODE TYPES (GRAPH CONNECTIVITY)\n"
             "   - PRIMARY NODES: 'person', 'organization'.\n"
             "   - SECONDARY NODES: 'location', 'event', 'concept', 'hobby', 'occupation' -> Create these for nouns/activities (e.g. 'Figure Skating', 'Python', 'Korea').\n"
             "     * CRITICAL: If multiple people share a hobby/job, it MUST be a Node to show the connection.\n"
             "   - ATTRIBUTES: 'value', 'state', 'trait' -> Keep simple adjectives (e.g. 'happy', 'tall') or status as attributes.\n\n"
             "   Rules for classification:\n"
             "   - IF entity is a Noun/Place/Event/Activity -> type='relation' (Node).\n"
             "   - IF entity is just an Adjective/Property -> type='attribute'.\n\n"
             "3) UNIVERSAL ATTRIBUTE SCHEMA (TIERS)\n"
             "   Map extracted info to these keys if possible (for Person entities):\n"
             "   - TIER 1 (Identity): 'name', 'type', 'role' (e.g. friend, coworker), 'gender' (optional).\n"
             "   - TIER 2 (Facts): 'age', 'birthday', 'location' (city/region), 'occupation' (job title), 'education' (school/major).\n"
             "   - TIER 3 (Social): 'is related to' (relationships), 'works at' (orgs). Note: These are usually RELATIONS (Nodes).\n"
             "   - TIER 4 (Preferences): 'likes', 'dislikes', 'skills', 'weaknesses'.\n"
             "   - TIER 5 (Personality): 'personality' (traits), 'habits'.\n"
             "   - TIER 6 (History): 'life event' (moved, started job). Note: Usually RELATIONS.\n"
             "   * Use these specific keys for attributes to keep the profile clean.\n"
             "   * Example: 'Miccel likes coding' -> (Miccel, likes, coding, attribute).\n"
             "   * Example: 'Miccel is a student' -> (Miccel, occupation, student, attribute).\n\n"
             "4) CANONICAL NAMES & NO REDUNDANCY\n"
             "   - Use canonical names (Minji, not Minji Kim).\n"
             "   - No redundant triplets.\n\n"
             "5) CHAINED LOCATIONS\n"
             "   - If 'Subject action Object at Location':\n"
             "       -> 'Subject action Object' (relation user->user/org) AND 'Object location is Location' (attribute).\n\n"
             "6) SCOPE\n"
             "   - Only use info in history/input. Return empty if none.\n"),
            ("user", "Chat History:\n{history}\n\nCurrent Input: {input}")
        ])
        
        # We pass history just in case, but coref is largely done. The Extraction prompt still helps for complex cases.
        history_text = "\n".join(state.get("chat_history", [])[-5:])
        chain = prompt | structured_llm
        result = chain.invoke({"input": input_text, "history": history_text})
        
        triplets_added = []
        
        # Determine the "Self" identity dynamically
        primary_user = self.memory.get_primary_user()
        # If no primary user exists yet (fresh graph), fallback to "User" or extraction results
        # However, we want to enforce consistency.
        # If the user says "I am Miccel", extraction will yield (Miccel, type, Miccel).
        # We only force-replace if we HAVE a primary user.
        target_identity = primary_user if primary_user else "User"

        for t in result.triplets:
            # 1. STRICT IDENTITY ENFORCEMENT
            # If the extraction resulted in 'I', 'Me', 'My', or 'User', force it to the Primary Identity
            if t.subject.lower() in ["i", "me", "my", "myself", "user"]:
                # If we have a known primary user, force it.
                # If we DON'T (target="User"), only force if the extracted name wasn't specific.
                # Actually, simplest rule: "User" -> Primary. "I" -> Primary.
                t.subject = target_identity
                t.subject_type = "person"
                
            if t.object_.lower() in ["i", "me", "my", "myself", "user"]:
                if t.object_type == "person":
                     t.object_ = target_identity

            # Special Case: If this is an identity declaration "User name is Miccel",
            # we might have (User, name, Miccel).
            # If we force subject="User", we get (User, name, Miccel).
            # Ideally we want (Miccel, name, Miccel). 
            # But the 'entity_linker' or 'coref' should handle "I am Miccel" -> "Miccel is Miccel".
            
            # Update Node Types first
            self.memory.add_node_type(t.subject, t.subject_type)
            if t.type == "relation":
                 self.memory.add_node_type(t.object_, t.object_type)

            if t.type == "attribute":
                self.memory.add_attribute(t.subject, t.predicate, t.object_)
                triplets_added.append(f"{t.subject} (attr: {t.predicate}) {t.object_}")
            else:
                meta = {"source": "user_input", "timestamp": datetime.datetime.now().isoformat()}
                self.memory.add_relation(t.subject, t.predicate, t.object_, metadata=meta)
                triplets_added.append(f"{t.subject} {t.predicate} {t.object_}")
            
        return {"extracted_triplets": triplets_added}

    def node_generate_storage_response(self, state: AgentState):
        """Step (Response): Generate a natural, conversational response after storing memories."""
        triplets = state.get("extracted_triplets", [])
        if not triplets:
            return {"response": "I didn't catch anything to remember there. Could you rephrase?"}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a warm, thoughtful, and intelligent second brain assistant. "
                       "You just stored the following new memories for the user.\n"
                       "Stored Facts: {triplets}\n"
                       "User Input: {input}\n\n"
                       "YOUR GOAL: Generate a short, natural, human-like response.\n"
                       "- ACKNOWLEDGE: Briefly confirm you understood, but DO NOT list the facts like a robot.\n"
                       "- ENGAGE: If it's a life event (graduation, move, job), say congratulations or ask how it is.\n"
                       "- CONNECT: If it's a preference (likes X), maybe ask a relevant follow-up.\n"
                       "- PROACTIVE: If a NEW PERSON is met, ask about their role/relationship AND ask how you feel about them.\n"
                       "- STATIC: If the user says 'nothing else' or just wants to store, simply say 'Got it' or 'Saved'.\n"
                       "- TONE: Friendly, professional, curious. Use 1-2 sentences max."),
            ("user", "Generate response.")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"triplets": ", ".join(triplets), "input": state["input"]}).content
        return {"response": response}

    def node_delete_memory(self, state: AgentState):
        """Handles deletion requests (Step 1: Ask Confirm, Step 2: Delete)."""
        intent = state.get("intent")
        
        if intent == "confirmed_delete":
            # Extract entity from LAST BOT MESSAGE
            # "Are you sure you want to permanently delete Minseok from your memory?"
            history = state.get("chat_history", [])
            
            # Find the confirmation message
            target_msg = ""
            for msg in reversed(history[-5:]):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if "Are you sure you want to permanently delete" in content:
                    target_msg = content
                    break
            
            # Simple parsing: assume format "delete [Entity] from"
            import re
            match = re.search(r"delete (.+?) from your memory", target_msg)
            if match:
                entity = match.group(1)
                success = self.memory.remove_node(entity)
                if success:
                    return {"response": f"Deleted {entity} from memory."}
                else:
                    return {"response": f"Could not find {entity} to delete."}
            else:
                return {"response": "I lost track of what to delete. Please ask again."}

        else: # Intent == 'delete'
            # Identify what to delete from current input
            entities = state.get("entities", [])
            if not entities:
                 return {"response": "I'm not sure what you want to delete. Please specify the name."}
            
            target = entities[0]
            # Verify existence
            if not self.memory._find_node(target):
                 return {"response": f"I don't have any memory of {target}."}
            
            # Return confirmation prompt
            return {"response": f"Are you sure you want to permanently delete {target} from your memory? This cannot be undone."}

    def node_general_chat(self, state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly and helpful personal assistant. The user just said something that isn't a memory or a question. Respond warmly and encourage them to share a memory."),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"input": state["rewritten_input"]})
        return {"response": response.content}

    def node_consolidate_memory(self, state: AgentState):
        """Step (Consolidate): Merge and clean up semantic duplicates in memory."""
        entities = state.get("entities", [])
        if not entities:
            print("[Consolidate] No entities to check.")
            return {}

        import json

        triplets = []
        raw_edges = [] 

        # 1. Gather Ego Network for recent entities
        for entity in entities:
             rels = self.memory.query_relations(entity)
             if rels:
                 for s, p, o, meta in rels:
                     # Flatten for LLM analysis
                     # We need the key to delete it later. meta usually doesn't have key in query_relations output 
                     # Wait, memory.py query_relations yields (s, p, o, meta). 
                     # I need the 'key' to delete specific edges. 
                     # I'll likely need to fetch edges directly or assume meta has it/I can find it.
                     # Let's inspect memory.py query_relations again or just iterate edges.
                     # It's safer to use the graph directly here since I'm the backend.
                     pass 

        # Direct Graph Access for Precision
        candidates = []
        
        for entity in entities:
             node = self.memory._find_node(entity)
             if not node: continue
             
             # Outgoing
             if self.memory.graph.has_node(node):
                 for neighbor in self.memory.graph.successors(node):
                     edge_data = self.memory.graph.get_edge_data(node, neighbor)
                     for key, attr in edge_data.items():
                         candidates.append({
                             "source": str(node),
                             "target": str(neighbor),
                             "predicate": attr.get('relation'),
                             "key": key
                         })
                         
             # Incoming (optional, but good for "is son of" vs "has son")
             # Leaving incoming out for V1 to keep it simple, focus on "Same Subject Duplication"
        
        if len(candidates) < 2:
            return {}

        # 2. Ask LLM to find duplicates
        candidates_json = json.dumps(candidates, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Knowledge Graph Curator. Your job is to identify SEMANTIC DUPLICATES in the list of connections.\n"
                       "Rules for Duplication:\n"
                       "1. Same Source + Same Target + Same Meaning Predicate = DUPLICATE.\n"
                       "   - Example: 'is brother of' AND 'brother of' -> Keep one.\n"
                       "   - Example: 'works at' AND 'is employed by' -> Keep one.\n"
                       "2. Same Source + Same Target + TIMESTAMP update -> Do NOT delete if unique history is needed, but if it looks like a retry, merge.\n"
                       "   - Generally, keep the shortest/cleanest predicate (e.g. 'works at' > 'is working as an employee at').\n\n"
                       "OUTPUT:\n"
                       "Return a JSON list of objects to DELETE. Structure: [{{'source': '...', 'target': '...', 'key': ...}}]\n"
                       "Return ONLY JSON. If no duplicates, return empty list []."),
            ("user", "Connections:\n{candidates_json}")
        ])
        
        chain = prompt | self.llm
        res = chain.invoke({"candidates_json": candidates_json})
        
        try:
            content = res.content.replace('```json', '').replace('```', '').strip()
            to_delete = json.loads(content)
            
            count = 0
            for item in to_delete:
                u = item.get("source")
                v = item.get("target")
                k = item.get("key")
                
                # Verify existence before deleting
                if self.memory.graph.has_edge(u, v, key=k):
                    self.memory.graph.remove_edge(u, v, key=k)
                    count += 1
            
            if count > 0:
                print(f"[Consolidate] Merged/Deleted {count} redundant edges.")
                self.memory.save_graph()
                
        except Exception as e:
            print(f"[Consolidate] Error parsing LLM response: {e}")
            
        return {}

    def node_query_planner(self, state: AgentState):
        """Step 0 (Query): Decompose complex queries."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Query Planner. Break down the user's request into a simple step-by-step plan if it is complex. If it is a simple question, just return '1. Answer the question'.\n"
                       "Return a newline-separated list of steps."),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        plan_text = chain.invoke({"input": state["rewritten_input"]}).content.strip()
        plan = [line.strip() for line in plan_text.split('\n') if line.strip()]
        print(f"[Planner] Plan: {plan}")
        return {"plan": plan}

    def node_retrieval(self, state: AgentState):
        """Step 1 (Query): Retrieve context for entities + expansion."""
        entities = state.get("entities", [])
        context_lines = []
        
        # 1. Initial Lookup
        for entity in entities:
             # Handle 'me' -> 'I' fallback
             if entity.lower() in ['me', 'my', 'myself']:
                 entity = 'I'
                 
             relations = self.memory.query_relations(entity)
             search_results = self.memory.search_memory(entity)
             
             if relations:
                 for s, p, o, meta in relations:
                     ts = f" [Time: {meta.get('timestamp')}]" if meta.get('timestamp') else ""
                     context_lines.append(f"{s} {p} {o}{ts}")
             if search_results:
                 context_lines.extend(search_results)
                 
        # Deduplicate
        context_lines = list(set(context_lines))
            
        if context_lines:
            initial_context = "\n".join(context_lines)
            
            # 2. Graph Expansion Loop
            expansion_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a research assistant. Based on the INITIAL CONTEXT and the USER QUERY, identify if we need to look up additional entities mentioned in the context to fully answer the question.\n"
                           "- Example: Query 'How do A and B know each other?' -> Context 'A works at Google', 'B works at Google'. -> No expansion needed (link found).\n"
                           "- Example: Query 'Who is A's boss?' -> Context 'A works at Google'. -> We need to search 'Google' to see who is the boss there. Return 'Google'.\n"
                           "- Return a COMMA-SEPARATED list of new entities to search. If none, return 'NONE'."),
                ("user", "User Query: {input}\n\nInitial Context:\n{context}")
            ])
            expand_chain = expansion_prompt | self.llm
            new_entities_str = expand_chain.invoke({"input": state["rewritten_input"], "context": initial_context}).content.strip()
            
            if "NONE" not in new_entities_str:
                new_entities = [e.strip().replace("'", "").replace('"', "") for e in new_entities_str.split(',')]
                for entity in new_entities:
                     if entity in entities: continue # Avoid re-search
                     
                     relations = self.memory.query_relations(entity)
                     if relations:
                         for s, p, o, meta in relations:
                             ts = f" [Time: {meta.get('timestamp')}]" if meta.get('timestamp') else ""
                             context_lines.append(f"{s} {p} {o}{ts}")

        final_context = list(set(context_lines))
        print(f"[Retrieval] Found {len(final_context)} facts.")
        return {"context": final_context}

    def node_reasoning(self, state: AgentState):
        """Step 2 (Query): Structured Reasoning."""
        context_text = "\n".join(state.get("context", []))
        if not context_text: 
            return {"reasoning_trace": {"analysis": "No context found.", "conclusion": "Unknown"}}

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Logical Reasoning Engine. Analyze the context to answer the query.\n"
                       "1. List DIRECT FACTS currently in the user's memory (e.g. 'Minji is friend of Miccel').\n"
                       "2. CHECK FOR INFERRED FACTS:\n"
                       "   - ONLY infer relationships that are *logically* certain (e.g. A is brother of B -> B is sibling of A).\n"
                       "   - DO NOT infer social relationships (e.g. Friends -> Sibling is FALSE).\n"
                       "   - DO NOT hallucinate facts not present in the context. If A's sibling is not mentioned, say 'Unknown'.\n"
                       "3. Identify CONTRADICTIONS if any (and resolve them based on recency if possible).\n"
                       "4. Formulate a final logical ANSWER.\n"
                       "Return a JSON object with keys: 'direct_facts', 'inferred_facts', 'contradictions', 'answer_logic'."),
            ("user", "Query: {input}\n\nContext:\n{context}")
        ])
        # We generally want structured output here, but let's use a standard LLM call and parse JSON for simplicity/flexibility
        chain = prompt | self.llm
        response = chain.invoke({"input": state["rewritten_input"], "context": context_text})
        
        # Simple trace storage (ideally we parse JSON, but string is fine for V1 trace)
        trace = {"raw_output": response.content}
        print(f"[Reasoning] {response.content[:100]}...")
        return {"reasoning_trace": trace}

    def node_answer(self, state: AgentState):
        """Step 3 (Query): Generate Final Response."""
        context_text = "\n".join(state.get("context", []))
        trace = state.get("reasoning_trace", {}).get("raw_output", "")
        
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a proactive personal memory assistant. Answer the question using the Analysis provided.\n"
                       "HYBRID SYNTHESIS STRATEGY:\n"
                       "1. MAIN ANSWER (World Knowledge): If the question asks for an opinion, facts, or assessment (e.g. 'Is X good?'), answer primarily using your own general knowledge.\n"
                       "2. PERSONAL CONTEXT (Memory): Check the provided context for any relevant connections.\n"
                       "   - If found, mentions them as 'Verified Personal Connections'.\n"
                       "   - Example: 'Yes, Korea University has a top-tier CS program... (World Knowledge). By the way, I recall that Minji studies there (Personal Memory).'\n"
                       "3. DO NOT FORCE logic. Do not say 'It is good BECAUSE Minji goes there'. Say 'It is good... AND Minji goes there'.\n"
                       "4. If the question is purely about personal memory (e.g. 'Where does Minji work?'), rely solely on context.\n"
                       "Be warm and natural. If no info is found and it's not a general question, say 'I don't know that yet'."),
            ("user", "User Query: {input}\n\nContext:\n{context}\n\nReasoning Analysis:\n{trace}")
        ])
        gen_chain = generation_prompt | self.llm
        response = gen_chain.invoke({"input": state["rewritten_input"], "context": context_text, "trace": trace})
        
        return {"response": response.content}

    def general_chat(self, state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly and helpful personal assistant. The user just said something that isn't a memory or a question. Respond warmly and encourage them to share a memory."),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"input": state["input"]})
        return {"response": response.content}

    def node_visualize_graph(self, state: AgentState):
        """Extracts entities to visualize from the user input."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are extracting entity names to visualize in a graph view.\n"
             "INPUT: User request (e.g. 'Visualize Minji', 'Show network of Miccel and Glenn').\n"
             "OUTPUT: A comma-separated list of strict entity names. If multiple, list them.\n"
             "Example: 'Visualize Minji' -> 'Minji'\n"
             "Example: 'Show me Miccel and LG' -> 'Miccel, LG'\n"),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({"input": state["input"]})
        
        # Parse comma separated results
        targets = [t.strip() for t in result.content.split(',') if t.strip()]
        
        # We also generate a confirmation response
        if not targets:
            response = "I couldn't identify who you want to visualize. "
        else:
            response = f"Visualizing network for: {', '.join(targets)}."
            
        return {"visualize_targets": targets, "response": response}

    def summarize_entity(self, entity: str) -> Dict[str, Any]:
        """Generates a profile summary for a given entity."""
        # 1. Get raw facts (structured) - Only use query_relations for clean profile data
        relations = self.memory.query_relations(entity)
        
        grouped_facts = {}
        context_lines = []
        entity_lower = entity.lower()
        seen = set()

        for s, p, o, meta in relations:
            # Handle List Values (User manual edits or future features)
            if isinstance(o, list):
                o_str = ", ".join([str(i) for i in o])
            else:
                o_str = str(o)

            # Filter self-loops/redundant nodes (e.g. Miccel is Miccel)
            if s.lower() == o_str.lower():
                continue

            # STRICT FILTER: Only include direct connections to the entity
            s_is_entity = (s.lower() == entity_lower)
            o_is_entity = (o_str.lower() == entity_lower)
            
            if not s_is_entity and not o_is_entity:
                continue
                
            # Deduplication Logic: Check for redundant "sibling" if "brother/sister" exists
            # We need to know all predicates for this connection to decide.
            # Since we are iterating, we strictly need a preprocessing step or a smart filter.
            # Simplified approach: If current pred is 'is sibling of', check if specific relation exists in the full list.
            
            if "sibling" in p.lower():
                # Check if specific relation exists for this pair (in EITHER direction)
                has_specific = False
                for s2, p2, o2, meta2 in relations:
                    # Robust list handling for inner loop too
                    o2_val = o2
                    if isinstance(o2, list):
                         o2_val = ", ".join([str(i) for i in o2])
                    else:
                         o2_val = str(o2)

                    # Check same direction matching s,o OR reverse direction matching o,s
                    same_dir = (s2 == s and o2_val == o_str)
                    rev_dir = (s2 == o_str and o2_val == s)
                    
                    if (same_dir or rev_dir) and ("brother" in p2.lower() or "sister" in p2.lower()):
                        has_specific = True
                        break
                if has_specific:
                     continue # Skip generic sibling relation
            
            # Create a unique key to prevent duplicates
            key = (s, p, o_str)
            if key in seen:
                continue
            seen.add(key)
            
            context_lines.append(f"{s} {p} {o_str}")
            
            # Grouping Logic
            # Direction handling: If s==entity, normal. If o==entity, incoming.
            
            label = p
            value = o_str
            
            if o_str.lower() == entity_lower:
                # Incoming: "Miccel is friends with Mytzka" (Viewing Mytzka)
                # Label: "is friends with (from)", Value: "Miccel"
                # To keep it cleaner/simpler for now, we'll append "(from ...)" to the label or similar.
                # Actually, for "is friends with", direction doesn't matter much.
                # For "studies at", "studies at (from)" works.
                label = f"{p} (from)"
                value = s
            
            # Clean up predicate
            # e.g. "is" -> "Details" or keep "is"
            # The user requested "Friends: minji, mytzka"
            # So if p="is friends with", label="Friends" would be nice, but we can't hardcode everything.
            # We'll stick to the predicate string for now.
            
            # Capitalize label
            label = label.capitalize()
            
            if label not in grouped_facts:
                grouped_facts[label] = []
            
            if value not in grouped_facts[label]:
                grouped_facts[label].append(value)
        
        if not context_lines:
             return {"bio": f"I don't have enough information about {entity} yet.", "facts": {}}

        context = "\n".join(context_lines)
        
        # 2. Summarize via LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a factual data summarizer. The user wants a summary of a specific entity based ONLY on the stored data.\n"
                       "1. Write a strictly factual profile summary (2-3 sentences).\n"
                       "2. FOCUS ONLY ON THE ENTITY. Do not mention friends of friends or unrelated details about other people linked to them.\n"
                       "3. DO NOT embellish. Be robotic but natural.\n"
                       "4. If no details exist, say 'No detailed information recorded.'"),
            ("user", "Entity: {entity}\n\nDirect Data Points:\n{context}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"entity": entity, "context": context})
        
        return {
            "bio": response.content,
            "facts": grouped_facts # Dictionary of grouped facts
        }

    def process_input(self, user_input: str, chat_history: List[str]):
        initial_state = {
            "input": user_input,
            "chat_history": chat_history,
            "rewritten_input": user_input,
            "entities": [],
            "intent": "chat",
            "plan": [],
            "context": [],
            "visualize_targets": [],
            "reasoning_trace": {"direct_facts": [], "inferred_facts": []},
            "response": ""
        }
        
        try:
            # Run the graph
            result = self.app.invoke(initial_state)
            
            # Extract visualization targets if present
            visualize_targets = result.get("visualize_targets", [])
            
            return {
                "response": result["response"],
                "visualize_targets": visualize_targets
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                return "⚠️ **Rate Limit Reached**: My brain is tired (Gemini Free Tier Limit). Please wait 1 minute and try again."
            raise e

