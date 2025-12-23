from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.memory import LocalMemoryStore
import json
import os

# Define the State
class AgentState(TypedDict):
    input: str
    intent: Literal["store", "query", "chat"]
    response: str
    extracted_triplets: List[List[str]] # [[Subject, Predicate, Object]]
    context: str
    chat_history: List[str]

# Define Structured Output for Extraction
class Triplet(BaseModel):
    subject: str = Field(description="The subject of the relationship or attribute")
    subject_type: Literal["person", "organization", "location", "object", "concept"] = Field(description="Type of the subject entity")
    predicate: str = Field(description="The relationship type or attribute name")
    object_: str = Field(description="The object of the relationship or attribute value")
    object_type: Literal["person", "organization", "location", "object", "concept", "value"] = Field(description="Type of the object entity (use 'value' for attributes)")
    type: Literal["relation", "attribute"] = Field(description="Type of information: 'relation' for node-to-node connection, 'attribute' for node property (e.g., adjectives)")

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
            # gpt-4o-mini is the cheapest and most capable small model from OpenAI
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        elif google_key:
            print("Using Google Gemini Model")
            # Using latest stable flash alias
            self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        else:
             print("CRITICAL WARNING: No API Key found (OPENAI_API_KEY or GOOGLE_API_KEY). Agent will fail.")
             # Dummy initialization to prevent crash before runtime
             self.llm = None

        # Build the Graph
        self.workflow = StateGraph(AgentState)
        
        # Add Nodes
        self.workflow.add_node("classify", self.classify_input)
        self.workflow.add_node("store_memory", self.store_memory)
        self.workflow.add_node("answer_query", self.answer_query)
        self.workflow.add_node("chat", self.general_chat)

        # Add Edges
        self.workflow.set_entry_point("classify")
        
        self.workflow.add_conditional_edges(
            "classify",
            lambda x: x["intent"],
            {
                "store": "store_memory",
                "query": "answer_query",
                "chat": "chat"
            }
        )
        
        self.workflow.add_edge("store_memory", END)
        self.workflow.add_edge("answer_query", END)
        self.workflow.add_edge("chat", END)

        self.app = self.workflow.compile()

    def classify_input(self, state: AgentState):
        # Combine history for context
        history_text = "\n".join(state.get("chat_history", [])[-3:]) # Last 3 Turns
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a rational memory assistant. Classify the user input into one of these categories:\n"
                       "- 'store': The user is telling you a fact to remember (e.g., 'Mom lives in New York', 'My name is John').\n"
                       "- 'query': The user is asking a question OR Following up on a previous question (e.g., 'Where does mom live?', 'Where is that?', 'And my dad?').\n"
                       "- 'chat': General greeting or non-memory interaction (e.g., 'Hello', 'How are you').\n"
                       "Return ONLY the category name.\n\n"
                       "Recent Chat History:\n{history}"),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"input": state["input"], "history": history_text})
        intent = response.content.strip().lower()
        
        # Fallback for safety
        if intent not in ["store", "query", "chat"]:
            intent = "chat"
            
        return {"intent": intent}

    def store_memory(self, state: AgentState):
        """Extracts entities and stores them in the graph."""
        structured_llm = self.llm.with_structured_output(ExtractionResult)
        
        # Combine history for context
        history_text = "\n".join(state.get("chat_history", [])[-3:])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract knowledge triples from the text.\n"
                       "- RESOLVE PRONOUNS & LOCATIONS using context. Look at the last turn of history.\n"
                       "  - 'She', 'He', 'It', 'They' MUST be replaced by the name of the person/object they refer to.\n"
                       "  - Handle contractions: 'shes' -> 'She is' -> 'Minji is'. 'hes' -> 'He is'.\n"
                       "- CLASSIFY TYPES: Detect the type of every entity (person, organization, location, object).\n"
                       "- CONCEPTS AS ATTRIBUTES: Abstract concepts, states, or activities (e.g. 'on a diet', 'in love', 'sick') should be ATTRIBUTES, not separate nodes.\n"
                       "- ATTRIBUTES: If the user describes a property (e.g., 'Minji is pretty'), classify as 'attribute'.\n"
                       "  - Example: 'Minji is pretty' -> Subject='Minji', SubType='person', Predicate='is', Object='pretty', ObjType='value', Type='attribute'.\n"
                       "  - Example: 'Mytzka is on a diet' -> Subject='Mytzka', SubType='person', Predicate='is on', Object='diet', ObjType='value', Type='attribute'.\n"
                       "- RELATIONS: If the user connects two entities.\n"
                       "  - Example: 'Minji goes to Korea University' -> Subject='Minji', SubType='person', Predicate='goes to', Object='Korea University', ObjType='organization', Type='relation'.\n"
                       "- CONSISTENCY: Use canonical names.\n"
                       "  - If 'Minji' exists, do not store 'Minji Kim' unless correcting.\n"
                       "- NO REDUNDANCY: Do not store multiple similar relationships for the same pair.\n"
                       "  - Example: If 'Miccel studies at Korea University', DO NOT also store 'Miccel is from Korea University'. Prefer the most specific one ('studies at').\n"
                       "  - Example: If 'Mom is 50', do not store 'Mom age is 50'.\n"
                       "- PERSON-CENTRIC: Prefer people as subjects.\n"
                       "- CHAINED LOCATIONS: If 'Subject action Object at Location', store 'Subject action Object' AND 'Object is located at Location'.\n"
                       "  - Example: 'I work at LG in Magok' -> 'I work at LG', 'LG is located at Magok'.\n"
                       "- POSSESSIVE RELATIONSHIPS: 'X is Y's Z' should be 'X is Z of Y'.\n"
                       "  - Example: 'Glenn is Miccel's mother' -> 'Glenn is mother of Miccel'.\n"
                       "- RECIPROCAL INFERENCE: Deduce inverse relationships.\n"
                       "  - If 'A is sister of B', store 'B is sibling of A' (or brother/sister if gender known).\n"
                       "- SPECIFICITY: Do not store generic 'sibling' if 'brother' or 'sister' is known."),
            ("user", "Chat History:\n{history}\n\nCurrent Input: {input}")
        ])
        chain = prompt | structured_llm
        result = chain.invoke({"input": state["input"], "history": history_text})
        
        triplets_added = []
        for t in result.triplets:
            # Update Node Types first
            self.memory.add_node_type(t.subject, t.subject_type)
            if t.type == "relation":
                 self.memory.add_node_type(t.object_, t.object_type)

            if t.type == "attribute":
                self.memory.add_attribute(t.subject, t.predicate, t.object_)
                triplets_added.append(f"{t.subject} (attr: {t.predicate}) {t.object_}")
            else:
                self.memory.add_relation(t.subject, t.predicate, t.object_)
                triplets_added.append(f"{t.subject} {t.predicate} {t.object_}")
            
        return {"response": f"I stored that: {', '.join(triplets_added)}", "extracted_triplets": triplets_added}

    def answer_query(self, state: AgentState):
        """Retrieves info and answers the question."""
        # 1. Identify key entities in the query (Simple keyword extraction or LLM)
        # For simplicity, let's just dump the graph context for now or search naive keywords
        # A better approach: Ask LLM which entities to look up
        
        # Quick heuristic: All nodes for now (small graph) OR heuristic keyword match
        # Let's do a simple full dump for context if small, otherwise keyword search
        # Since this is a specialized agent, let's ask LLM "What entity is this about?"
        
        # 1. Identify key entities in the query
        # 1. Identify key entities in the query
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Identify all key entities to search for in this query. Return them as a COMMA-SEPARATED list.\n"
                       "- If the user asks about themselves (e.g., 'Where do *I* live?', 'My name'), return 'I'.\n"
                       "- If asking about a location (e.g., 'What is in the kitchen?'), return the location ('Kitchen').\n"
                       "- Simplify to the core entities (e.g., 'mom's birthday' -> 'Mom').\n"
                       "- Example: 'Difference between Miccel and Minji' -> 'Miccel, Minji'\n"
                       "- RESOLVE PRONOUNS: If the query uses pronouns like 'he', 'she', 'they', or 'it', refer to the CHAT HISTORY to identify who is being discussed.\n"
                       "- Example: History='Tell me about Miccel.', Query='Where does he work?' -> Return 'Miccel'."),
            ("user", "Chat History:\n{history}\n\nCurrent Query: {input}")
        ])
        chain = prompt | self.llm
        entities_str = chain.invoke({"input": state["input"], "history": state.get("history", [])}).content.strip()
        
        # Parse entities
        entities = [e.strip().replace("'", "").replace('"', "") for e in entities_str.split(',')]
        
        context_lines = []
        for entity in entities:
             # Handle 'me' -> 'I' fallback
             if entity.lower() in ['me', 'my', 'myself']:
                 entity = 'I'
                 
             # Get context for each entity
             # We use depth=1 for multi-entity to avoid context explosion, or keep depth=2 if graph is small.
             # Let's use standard logic.
             
             relations = self.memory.query_relations(entity)
             search_results = self.memory.search_memory(entity)
             
             if relations:
                 context_lines.extend([f"{s} {p} {o}" for s, p, o in relations])
             if search_results:
                 context_lines.extend(search_results)
                 
        # Deduplicate
        context_lines = list(set(context_lines))
            
        if context_lines:
            initial_context = "\n".join(list(set(context_lines)))
            
            # 2. Graph Expansion: Check if we need to follow links
            expansion_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a research assistant. Based on the INITIAL CONTEXT and the USER QUERY, identify if we need to look up additional entities mentioned in the context to fully answer the question.\n"
                           "- Example: Query 'How do A and B know each other?' -> Context 'A works at Google', 'B works at Google'. -> No expansion needed (link found).\n"
                           "- Example: Query 'Who is A's boss?' -> Context 'A works at Google'. -> We need to search 'Google' to see who is the boss there. Return 'Google'.\n"
                           "- Return a COMMA-SEPARATED list of new entities to search. If none, return 'NONE'."),
                ("user", "User Query: {input}\n\nInitial Context:\n{context}")
            ])
            expand_chain = expansion_prompt | self.llm
            new_entities_str = expand_chain.invoke({"input": state["input"], "context": initial_context}).content.strip()
            
            if "NONE" not in new_entities_str:
                new_entities = [e.strip().replace("'", "").replace('"', "") for e in new_entities_str.split(',')]
                for entity in new_entities:
                     # Avoid re-searching known entities
                     if entity in entities: 
                         continue
                     
                     relations = self.memory.query_relations(entity)
                     if relations:
                         context_lines.extend([f"{s} {p} {o}" for s, p, o in relations])
                     # We can skip search_memory for expansion to avoid noise, relations are usually what we want for traversing.

        # Deduplicate Final Context
        final_context = "\n".join(list(set(context_lines))) if context_lines else "No direct information found."

        # 3. Generate Answer with Deduction
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a proactive personal memory assistant. Answer the question using the provided Context.\n"
                       "1. ANALYZE relationships. If A and B share a location (e.g. both study at KU), deduce they are likely schoolmates/colleagues.\n"
                       "2. CHAIN facts. If 'Miccel works at LG' and 'LG is in Magok', answer 'Miccel works in Magok (at LG)'.\n"
                       "3. If the context contains other *relevant* details (e.g., if asked about a person, briefly mention their location/job), mention them.\n"
                       "4. STRICT RELEVANCE: If the user asks about a specific topic (e.g. 'work'), DISCARD context facts about unrelated topics (e.g. 'school', 'age', 'friends'). Only mention cross-domain facts if they directly answer the question.\n"
                       "5. ENRICH answers with general world knowledge. If the context mentions a well-known entity (e.g. 'LG'), you MAY add a brief description (e.g. 'a major electronics conglomerate') if relevant.\n"
                       "  - Note: Prioritize user Context, but use general knowledge to fill gaps or add flavor.\n"
                       "If the context is empty, say 'I don't know that yet'."),
            ("user", "Context:\n{context}\n\nQuestion: {input}")
        ])
        gen_chain = generation_prompt | self.llm
        response = gen_chain.invoke({"context": final_context, "input": state["input"]})
        
        return {"response": response.content, "context": final_context}

    def general_chat(self, state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly and helpful personal assistant. The user just said something that isn't a memory or a question. Respond warmly and encourage them to share a memory."),
            ("user", "{input}")
        ])
        chain = prompt | self.llm
        response = chain.invoke({"input": state["input"]})
        return {"response": response.content}

    def summarize_entity(self, entity: str) -> Dict[str, Any]:
        """Generates a profile summary for a given entity."""
        # 1. Get raw facts (structured) - Only use query_relations for clean profile data
        relations = self.memory.query_relations(entity)
        
        grouped_facts = {}
        context_lines = []
        entity_lower = entity.lower()
        seen = set()

        for s, p, o in relations:
            # Filter self-loops/redundant nodes (e.g. Miccel is Miccel)
            if s.lower() == o.lower():
                continue

            # STRICT FILTER: Only include direct connections to the entity
            s_is_entity = (s.lower() == entity_lower)
            o_is_entity = (o.lower() == entity_lower)
            
            if not s_is_entity and not o_is_entity:
                continue
                
            # Deduplication Logic: Check for redundant "sibling" if "brother/sister" exists
            # We need to know all predicates for this connection to decide.
            # Since we are iterating, we strictly need a preprocessing step or a smart filter.
            # Simplified approach: If current pred is 'is sibling of', check if specific relation exists in the full list.
            
            if "sibling" in p.lower():
                # Check if specific relation exists for this pair (in EITHER direction)
                has_specific = False
                for s2, p2, o2 in relations:
                    # Check same direction matching s,o OR reverse direction matching o,s
                    same_dir = (s2 == s and o2 == o)
                    rev_dir = (s2 == o and o2 == s)
                    
                    if (same_dir or rev_dir) and ("brother" in p2.lower() or "sister" in p2.lower()):
                        has_specific = True
                        break
                if has_specific:
                     continue # Skip generic sibling relation
            
            # Create a unique key to prevent duplicates
            key = (s, p, o)
            if key in seen:
                continue
            seen.add(key)
            
            context_lines.append(f"{s} {p} {o}")
            
            # Grouping Logic
            # Direction handling: If s==entity, normal. If o==entity, incoming.
            
            label = p
            value = o
            
            if o.lower() == entity_lower:
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
            "intent": "chat",
            "response": "",
            "extracted_triplets": [],
            "context": "",
            "chat_history": chat_history
        }
        try:
            result = self.app.invoke(initial_state)
            return result["response"]
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                return "⚠️ **Rate Limit Reached**: My brain is tired (Gemini Free Tier Limit). Please wait 1 minute and try again."
            raise e
