import os
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Tuple, Optional, Union
import json
import re
import numpy as np
import datetime
import pickle
from pathlib import Path

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Tool imports
import requests
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools import Tool
from langchain_core.tools import tool

# FAISS and embedding imports
from sentence_transformers import SentenceTransformer
import faiss

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# Step 1: Define the state structure for the HR Helper
class HRState(TypedDict):
    messages: Annotated[Sequence[Any], "The messages in the conversation so far"]
    hiring_needs: Annotated[List[str], "List of positions to be hired"]
    current_position: Annotated[str, "The position currently being worked on"]
    position_details: Annotated[Dict[str, Dict[str, Any]], "Details for each position (skills, budget, etc.)"]
    job_descriptions: Annotated[Dict[str, str], "Job descriptions for each position"]
    hiring_plans: Annotated[Dict[str, str], "Hiring plans/checklist for each position"]
    clarification_complete: Annotated[bool, "Whether clarification questions have been asked"]
    stage: Annotated[str, "Current stage of the HR process (clarify, job_desc, plan, complete)"]
    interaction_count: Annotated[int, "Count of interactions to prevent infinite loops"]
    checklists: Annotated[Dict[str, List[Dict[str, Any]]], "Custom checklists for each position"]
    draft_emails: Annotated[Dict[str, Dict[str, str]], "Email drafts for different purposes"]
    search_results: Annotated[Dict[str, List[Dict[str, str]]], "Stored search results for reference"]
    faiss_index: Annotated[Any, "FAISS index for vector search"]
    document_store: Annotated[List[Dict[str, Any]], "Store for documents used in RAG"]
    session_id: Annotated[str, "Unique identifier for this session"]
    session_state: Annotated[Dict[str, Any], "Persistent state across session steps"]

# Step 2: Create the LLM - add your API key here or set as environment variable
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 3: Initialize FAISS and embeddings for RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")
STATE_DIR = os.path.join(DATA_DIR, "state")
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# Try to load the embedding model
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    print(f"Initialized embedding model: {EMBEDDING_MODEL}, dimension: {EMBEDDING_DIM}")
    RAG_ENABLED = True
except Exception as e:
    print(f"Warning: Could not initialize embedding model: {str(e)}")
    print("Fallback to basic mode without RAG")
    embedding_model = None
    EMBEDDING_DIM = 384  # Default for MiniLM-L6-v2
    RAG_ENABLED = False

# State Management Functions
def generate_session_id():
    """Generate a unique session ID."""
    return f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

def save_session_state(session_id: str, state_data: Dict[str, Any]) -> bool:
    """Save session state to disk."""
    try:
        state_path = os.path.join(STATE_DIR, f"{session_id}.json")
        with open(state_path, 'w') as f:
            # Convert to JSON-serializable format
            serializable_state = {}
            for key, value in state_data.items():
                if key not in ['faiss_index', 'messages']:  # Skip non-serializable items
                    serializable_state[key] = value
            
            json.dump(serializable_state, f, indent=2, default=str)
        
        # Save messages separately
        if 'messages' in state_data:
            messages_path = os.path.join(STATE_DIR, f"{session_id}_messages.pkl")
            with open(messages_path, 'wb') as f:
                pickle.dump(state_data.get('messages', []), f)
        
        print(f"Saved session state for {session_id}")
        return True
    except Exception as e:
        print(f"Error saving session state: {str(e)}")
        return False

def load_session_state(session_id: str) -> Dict[str, Any]:
    """Load session state from disk."""
    try:
        state_path = os.path.join(STATE_DIR, f"{session_id}.json")
        messages_path = os.path.join(STATE_DIR, f"{session_id}_messages.pkl")
        
        state_data = {}
        
        # Load main state data
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state_data = json.load(f)
        
        # Load messages if available
        if os.path.exists(messages_path):
            with open(messages_path, 'rb') as f:
                state_data['messages'] = pickle.load(f)
        
        print(f"Loaded session state for {session_id}")
        return state_data
    except Exception as e:
        print(f"Error loading session state: {str(e)}")
        return {}

def list_available_sessions() -> List[Dict[str, Any]]:
    """List all available sessions with basic info."""
    sessions = []
    for file in os.listdir(STATE_DIR):
        if file.endswith('.json'):
            session_id = file.replace('.json', '')
            try:
                with open(os.path.join(STATE_DIR, file), 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': session_id,
                        'position': data.get('current_position', 'Unknown'),
                        'stage': data.get('stage', 'Unknown'),
                        'time': data.get('timestamp', 'Unknown')
                    })
            except Exception as e:
                print(f"Error reading session {session_id}: {str(e)}")
    
    return sorted(sessions, key=lambda s: s.get('time', ''), reverse=True)

def get_session_history(position: str = None) -> List[Dict[str, Any]]:
    """Get history of sessions for a specific position or all sessions."""
    all_sessions = list_available_sessions()
    
    if position:
        return [s for s in all_sessions if s.get('position', '').lower() == position.lower()]
    else:
        return all_sessions

# RAG Utility Functions
def init_faiss_index():
    """Initialize or load FAISS index."""
    try:
        index_path = os.path.join(VECTOR_DIR, "hr_index.faiss")
        docs_path = os.path.join(VECTOR_DIR, "hr_docs.json")
        
        # Create new index if none exists
        if not os.path.exists(index_path):
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            documents = []
            print(f"Created new FAISS index with dimension {EMBEDDING_DIM}")
            return index, documents
        
        # Load existing index and documents
        index = faiss.read_index(index_path)
        
        with open(docs_path, "r") as f:
            documents = json.load(f)
        
        print(f"Loaded FAISS index with {index.ntotal} vectors and {len(documents)} documents")
        return index, documents
    except Exception as e:
        print(f"Error initializing FAISS index: {str(e)}")
        # Return empty index and documents as fallback
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        return index, []

def save_faiss_index(index, documents):
    """Save FAISS index and documents to disk."""
    try:
        os.makedirs(VECTOR_DIR, exist_ok=True)
        index_path = os.path.join(VECTOR_DIR, "hr_index.faiss")
        docs_path = os.path.join(VECTOR_DIR, "hr_docs.json")
        
        # Save the FAISS index
        faiss.write_index(index, index_path)
        
        # Save the documents
        with open(docs_path, "w") as f:
            json.dump(documents, f, indent=2)
        
        print(f"Saved FAISS index with {index.ntotal} vectors and {len(documents)} documents")
        return True
    except Exception as e:
        print(f"Error saving FAISS index: {str(e)}")
        return False

def add_to_faiss(index, documents, text, metadata):
    """Add a document to the FAISS index."""
    if not RAG_ENABLED or embedding_model is None:
        return index, documents
    
    try:
        # Add timestamp to metadata
        metadata = {
            **metadata,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Create embedding
        embedding = embedding_model.encode([text])[0]
        vector = np.array([embedding]).astype('float32')
        
        # Add to index
        index.add(vector)
        
        # Add to documents store
        documents.append({
            "text": text,
            "metadata": metadata
        })
        
        print(f"Added document to FAISS index. Total documents: {len(documents)}")
        return index, documents
    except Exception as e:
        print(f"Error adding to FAISS: {str(e)}")
        return index, documents

def search_faiss(index, documents, query, k=3):
    """Search FAISS index for similar documents."""
    if not RAG_ENABLED or embedding_model is None or len(documents) == 0:
        return []
    
    try:
        # Create embedding for query
        embedding = embedding_model.encode([query])[0]
        vector = np.array([embedding]).astype('float32')
        
        # Limit k to the number of documents
        k = min(k, len(documents))
        if k == 0:
            return []
        
        # Search index
        distances, indices = index.search(vector, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(documents):
                results.append({
                    "document": documents[idx],
                    "distance": float(distances[0][i]),
                    "similarity": 1.0 / (1.0 + float(distances[0][i]))
                })
        
        return results
    except Exception as e:
        print(f"Error searching FAISS: {str(e)}")
        return []

def extract_hiring_needs(state: HRState) -> HRState:
    """Extract the position that needs to be hired from the user's input using LLM."""
    print("Extracting hiring needs with LLM...")
    
    # Create a session ID if not present
    session_id = state.get("session_id", generate_session_id())
    
    # Get the user's initial message
    user_input = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            user_input = message.content
            break
    
    print(f"User input: '{user_input}'")
    
    # Initialize FAISS for RAG
    faiss_index, document_store = init_faiss_index()
    
    # Skip command line arguments or Jupyter kernel paths
    if user_input.startswith("-") or ".json" in user_input:
        position = "Software Developer"
        print(f"Input appears to be a command line argument. Using default position: {position}")
    else:
        # Check if we have similar requests in FAISS (RAG)
        similar_docs = []
        if RAG_ENABLED and embedding_model is not None:
            similar_docs = search_faiss(faiss_index, document_store, user_input, k=2)
            
        # If we found similar requests, use them to help determine the position
        if similar_docs:
            print(f"Found {len(similar_docs)} similar requests in vector store")
            # Check if the most similar document can help determine position
            top_doc = similar_docs[0]["document"]
            top_metadata = top_doc.get("metadata", {})
            if "position" in top_metadata and similar_docs[0]["similarity"] > 0.8:
                position = top_metadata["position"]
                print(f"Using position from similar document with {similar_docs[0]['similarity']:.2f} similarity: {position}")
            else:
                # Use LLM to extract position
                position = extract_position_with_llm(user_input)
        else:
            # Use LLM to extract the position
            position = extract_position_with_llm(user_input)
    
    # Add user request to FAISS for future reference
    if RAG_ENABLED and embedding_model is not None:
        faiss_index, document_store = add_to_faiss(
            faiss_index, 
            document_store, 
            user_input, 
            {"type": "request", "position": position}
        )
        save_faiss_index(faiss_index, document_store)
    
    # Initialize position details
    position_details = {
        position: {
            "skills": [],
            "budget": None,
            "timeline": None,
            "experience_level": None,
            "location": None
        }
    }
    
    # Initialize session state
    session_state = {
        "start_time": datetime.datetime.now().isoformat(),
        "position": position,
        "completed_steps": ["extract_hiring_needs"],
        "position_history": [position],
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save the initial session state
    save_session_state(session_id, {
        "current_position": position,
        "stage": "clarify",
        "session_state": session_state,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    # Update the state
    return {
        **state,
        "hiring_needs": [position],
        "current_position": position,
        "position_details": position_details,
        "job_descriptions": {position: ""},
        "hiring_plans": {position: ""},
        "clarification_complete": False,
        "stage": "clarify",
        "interaction_count": 0,
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_id": session_id,
        "session_state": session_state,
        "messages": state["messages"] + [
            AIMessage(content=f"I'll help you hire for the {position} position. Let's start by gathering some details.")
        ]
    }

def extract_position_with_llm(user_input: str) -> str:
    """Extract position using LLM."""
    extraction_prompt = f"""
    Extract the job position the user wants to hire for from the text.
    If multiple positions are mentioned, identify the main position they're focused on.
    If no specific position is mentioned, infer the most likely position based on context.

    Text: "{user_input}"

    Return your answer in this exact format:
    {{
        "position": "job position name" // just the position title, keep it concise
    }}
    """
    
    parser = JsonOutputParser()
    extraction_chain = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert at extracting hiring information from text."),
        HumanMessage(content=extraction_prompt)
    ]) | llm | parser
    
    try:
        # Extract the position using the LLM
        extracted_info = extraction_chain.invoke({})
        position = extracted_info.get("position", "Software Developer")
        print(f"LLM extracted position: {position}")
        return position
    except Exception as e:
        print(f"Error extracting position with LLM: {str(e)}")
        return "Software Developer"

def ask_clarifying_questions(state: HRState) -> HRState:
    """Ask a clarifying question using LLM based on missing details."""
    print("Asking clarifying question...")

    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {}).copy()
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})

    # Determine missing fields
    missing_info = []
    if not position_details.get("skills"):
        missing_info.append("skills or requirements")
    if not position_details.get("budget"):
        missing_info.append("budget or salary range")
    if not position_details.get("experience_level"):
        missing_info.append("required experience level")
    if not position_details.get("location"):
        missing_info.append("work location or arrangement")
    if not position_details.get("timeline"):
        missing_info.append("hiring timeline")

    conversation_text = "\n".join([
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
        for m in state["messages"][-6:]
    ])

    # RAG enhancement: Look for similar positions to inform our questions
    rag_context = ""
    if RAG_ENABLED and embedding_model is not None:
        query = f"{current_position} position requirements"
        similar_docs = search_faiss(faiss_index, document_store, query, k=2)
        
        if similar_docs:
            rag_context = "Based on similar positions in our database, consider these details:\n\n"
            for i, result in enumerate(similar_docs, 1):
                doc = result["document"]
                if doc["metadata"].get("type") in ["job_description", "clarification"]:
                    rag_context += f"Example {i}: {doc['text'][:200]}...\n\n"

    # Prompt LLM to ask next question
    prompt = f"""
You're helping to collect information for a job posting for: {current_position}.

Missing info: {', '.join(missing_info) if missing_info else 'None'}

Conversation so far:
{conversation_text}

{rag_context if rag_context else ""}

Generate ONE short question to clarify the most important missing information.
If nothing is missing, respond only with [DONE].
"""

    chain = ChatPromptTemplate.from_messages([
        SystemMessage(content="You help HR teams gather complete job posting information."),
        HumanMessage(content=prompt)
    ]) | llm

    try:
        result = chain.invoke({})
        question = result.content.strip()

        if question.upper() == "[DONE]":
            print("LLM indicated no further questions.")
            
            # Update session state
            session_state["completed_steps"] = session_state.get("completed_steps", []) + ["clarification"]
            session_state["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save updated state
            if session_id:
                save_session_state(session_id, {
                    "current_position": current_position,
                    "stage": "job_desc",
                    "session_state": session_state,
                    "position_details": state["position_details"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
            
            return {
                **state,
                "clarification_complete": True,
                "stage": "job_desc",
                "session_state": session_state,
                "messages": state["messages"] + [
                    AIMessage(content=f"Thanks! I think I have enough information to write a job description for the {current_position}.")
                ]
            }
        else:
            print(f"LLM generated question: {question}")
            
            # Update session state
            session_state["last_question"] = question
            session_state["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save updated state
            if session_id:
                save_session_state(session_id, {
                    "current_position": current_position,
                    "stage": "clarify",
                    "session_state": session_state,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            
            return {
                **state,
                "session_state": session_state,
                "messages": state["messages"] + [AIMessage(content=question)]
            }

    except Exception as e:
        print(f"Fallback question due to LLM error: {e}")
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"Could you tell me more about the {current_position} position, like key skills, budget, or experience?")
            ]
        }


def process_clarification_responses(state: HRState) -> HRState:
    """Process user's latest answer to a clarifying question."""
    print("Processing clarification response...")

    latest_response = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    current_position = state["current_position"]
    current_details = state["position_details"].get(current_position, {}).copy()
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})

    # Prompt LLM to extract structured data
    prompt = f"""
You're helping to gather hiring information for the position: {current_position}.

The user's response:
"{latest_response}"

Extract the following if mentioned:
- skills (as a list)
- budget (salary range)
- experience level
- work location/arrangement
- hiring timeline

Return as JSON like:
{{
  "skills": ["skill1", "skill2"],
  "budget": "value",
  "experience_level": "value",
  "location": "value",
  "timeline": "value"
}}
Only include keys that were actually mentioned.
"""

    parser = JsonOutputParser()
    chain = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a structured information extractor."),
        HumanMessage(content=prompt)
    ]) | llm | parser

    try:
        extracted = chain.invoke({})

        for key, value in extracted.items():
            if value:
                current_details[key] = value
                print(f"Extracted {key}: {value}")

    except Exception as e:
        print(f"Failed to parse clarification: {e}")

    # Store this in FAISS for future reference
    if RAG_ENABLED and embedding_model is not None:
        # Create document with extracted details
        clarification_doc = {
            "type": "clarification",
            "position": current_position, 
            "extracted": json.dumps(extracted) if 'extracted' in locals() else "{}"
        }
        
        # Add to FAISS
        faiss_index, document_store = add_to_faiss(
            faiss_index,
            document_store,
            latest_response,
            clarification_doc
        )
        
        # Save index periodically
        save_faiss_index(faiss_index, document_store)
    
    # Update session state
    session_state["interaction_count"] = session_state.get("interaction_count", 0) + 1
    session_state["last_response"] = latest_response
    session_state["timestamp"] = datetime.datetime.now().isoformat()
    
    # Track extracted details in session state
    session_state["extracted_details"] = session_state.get("extracted_details", {})
    if 'extracted' in locals():
        for key, value in extracted.items():
            if value:
                session_state["extracted_details"][key] = value
    
    # Save session state
    if session_id:
        save_session_state(session_id, {
            "current_position": current_position,
            "stage": "clarify",
            "position_details": {
                **state["position_details"],
                current_position: current_details
            },
            "session_state": session_state,
            "timestamp": datetime.datetime.now().isoformat()
        })

    return {
        **state,
        "position_details": {
            **state["position_details"],
            current_position: current_details
        },
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_state": session_state,
        "messages": state["messages"] + [
            AIMessage(content="Thanks, I've noted that. Let me see if I need to ask anything else.")
        ]
    }

def should_continue_clarifying(state: HRState) -> str:
    current_position = state["current_position"]
    details = state["position_details"].get(current_position, {})
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})

    filled = sum(bool(v) for v in details.values())
    if filled >= 3:  # or however many you consider "enough"
        print("Enough information gathered.")
        
        # Update session state
        if session_state:
            session_state["clarification_complete"] = True
            session_state["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save state
            if session_id:
                save_session_state(session_id, {
                    "current_position": current_position,
                    "stage": "job_desc",  # Moving to next stage
                    "position_details": state["position_details"],
                    "session_state": session_state,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        return "move_to_job_desc"
    else:
        print("Still missing info. Continue clarifying.")
        
        # Update session state
        if session_state:
            session_state["clarification_complete"] = False
            session_state["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save state
            if session_id:
                save_session_state(session_id, {
                    "current_position": current_position,
                    "stage": "clarify",  # Staying in clarification stage
                    "position_details": state["position_details"],
                    "session_state": session_state,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        return "continue_clarifying"

def human_input(state: HRState) -> HRState:
    """Function to collect human input in the workflow.
    
    This function pauses the workflow execution and prompts the user for input
    through the command line. The input is then added to the state as a HumanMessage.
    """
    current_position = state["current_position"]
    session_id = state.get("session_id")
    
    # Get the last question asked by the system
    last_question = next((m.content for m in reversed(state["messages"]) 
                         if isinstance(m, AIMessage)), "")
    
    print("\n" + "-"*50)
    print(f"HR Helper is asking about the {current_position} position:")
    print(f"{last_question}")
    print("-"*50)
    
    # Get user input from command line
    try:
        user_response = input("\nYour response: ")
        
        # Check if the user provided any input
        if not user_response.strip():
            print("No input detected. Please provide a response.")
            return human_input(state)  # Recursively call until we get input
            
        print(f"Response received: {user_response[:50]}..." if len(user_response) > 50 else f"Response received: {user_response}")
        
    except KeyboardInterrupt:
        print("\nInput interrupted. Using default response.")
        user_response = "I need to think about this more."
    except Exception as e:
        print(f"\nError receiving input: {str(e)}. Using default response.")
        user_response = "I'm not sure how to answer that right now."
    
    # Save temporary state with this input
    if session_id:
        current_messages = state["messages"] + [HumanMessage(content=user_response)]
        save_session_state(session_id, {
            "messages": current_messages,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    # Return updated state with the human message added
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_response)]
    }

def create_job_description(state: HRState) -> HRState:
    """Create a job description for the current position."""
    print("Creating job description...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})
    
    # Ensure we have some skills
    if not position_details.get("skills"):
        position_details = position_details.copy()
        position_details["skills"] = ["relevant technical skills", "problem-solving abilities", "communication skills"]
    
    # RAG enhancement: Look for similar job descriptions
    similar_job_descriptions = ""
    if RAG_ENABLED and embedding_model is not None:
        query = f"job description {current_position} {' '.join(position_details.get('skills', []))}"
        similar_docs = search_faiss(faiss_index, document_store, query, k=2)
        
        if similar_docs:
            similar_job_descriptions = "Here are some similar job descriptions for reference:\n\n"
            for i, result in enumerate(similar_docs, 1):
                doc = result["document"]
                if doc["metadata"].get("type") == "job_description":
                    similar_job_descriptions += f"Example {i} ({result['similarity']:.2f} similarity):\n{doc['text'][:300]}...\n\n"
    
    try:
        job_desc_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an experienced HR professional and technical recruiter. 
            Create a compelling and detailed job description for the specified position.
            Include sections for:
            - About the role
            - Responsibilities
            - Requirements
            - Nice-to-have skills
            - Benefits
            - Application process
            
            Make the description professional, engaging, and formatted in Markdown."""),
            HumanMessage(content=f"""Position: {current_position}
            Details:
            - Skills: {', '.join(position_details.get('skills', ['']))}
            - Budget/Salary: {position_details.get('budget', 'Competitive')}
            - Timeline: {position_details.get('timeline', 'As soon as possible')}
            - Experience Level: {position_details.get('experience_level', 'Not specified')}
            - Location: {position_details.get('location', 'Not specified')}
            
            {similar_job_descriptions if similar_job_descriptions else ""}
            
            Please create a complete job description using this information.""")
        ])
        
        job_desc_chain = job_desc_prompt | llm
        job_desc_response = job_desc_chain.invoke({})
        
        job_description = job_desc_response.content
    except Exception as e:
        print(f"Error creating job description: {str(e)}")
        # Provide a generic job description if there's an error
        job_description = f"""
            # {current_position}

            ## About the Role
            We are seeking a talented {current_position} to join our team. This role offers an opportunity to work on exciting projects and make a significant impact.

            ## Responsibilities
            - Collaborate with team members to achieve project goals
            - Apply technical expertise to solve complex problems
            - Stay updated with industry trends and technologies

            ## Requirements
            - Experience in relevant technologies and methodologies
            - Strong problem-solving and analytical skills
            - Excellent communication and teamwork abilities

            ## Benefits
            - Competitive salary
            - Professional development opportunities
            - Flexible work arrangements

            ## How to Apply
            Please submit your resume and a brief cover letter outlining your qualifications.
            """
    
    # Store job description in FAISS for future reference
    if RAG_ENABLED and embedding_model is not None:
        faiss_index, document_store = add_to_faiss(
            faiss_index,
            document_store,
            job_description,
            {
                "type": "job_description",
                "position": current_position,
                "skills": position_details.get("skills", [])
            }
        )
        save_faiss_index(faiss_index, document_store)
    
    # Update session state
    session_state["completed_steps"] = session_state.get("completed_steps", []) + ["job_description"]
    session_state["timestamp"] = datetime.datetime.now().isoformat()
    
    # Save job description to file
    job_desc_dir = os.path.join(DATA_DIR, "job_descriptions")
    os.makedirs(job_desc_dir, exist_ok=True)
    job_desc_path = os.path.join(job_desc_dir, f"{current_position.replace(' ', '_').lower()}_job_description.md")
    try:
        with open(job_desc_path, 'w') as f:
            f.write(job_description)
        print(f"Saved job description to {job_desc_path}")
    except Exception as e:
        print(f"Error saving job description to file: {str(e)}")
    
    # Save session state
    if session_id:
        save_session_state(session_id, {
            "current_position": current_position,
            "stage": "plan",  # Moving to next stage
            "job_descriptions": {
                **state.get("job_descriptions", {}),
                current_position: job_description
            },
            "session_state": session_state,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    # Update the job description in the state
    job_descriptions = state["job_descriptions"].copy()
    job_descriptions[current_position] = job_description
    
    # Update the state
    return {
        **state,
        "job_descriptions": job_descriptions,
        "stage": "plan",
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_state": session_state,
        "messages": state["messages"] + [
            AIMessage(content=f"Here's a job description for the {current_position} position:\n\n{job_description}\n\nNext, I'll create a hiring plan for this position.")
        ]
    }

def create_hiring_plan(state: HRState) -> HRState:
    """Create a hiring plan/checklist for the current position."""
    print("Creating hiring plan...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})
    
    # RAG enhancement: Look for similar hiring plans
    similar_hiring_plans = ""
    if RAG_ENABLED and embedding_model is not None:
        query = f"hiring plan {current_position}"
        similar_docs = search_faiss(faiss_index, document_store, query, k=2)
        
        if similar_docs:
            similar_hiring_plans = "Here are some similar hiring plans for reference:\n\n"
            for i, result in enumerate(similar_docs, 1):
                doc = result["document"]
                if doc["metadata"].get("type") == "hiring_plan":
                    similar_hiring_plans += f"Example {i} ({result['similarity']:.2f} similarity):\n{doc['text'][:300]}...\n\n"
    
    try:
        plan_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an experienced HR professional. 
            Create a detailed hiring plan/checklist for the specified position.
            Include sections for:
            - Sourcing candidates
            - Screening process
            - Interview stages
            - Assessment methods
            - Decision timeline
            - Onboarding steps
            
            Format the plan as a structured Markdown checklist with main sections and sub-items."""),
            HumanMessage(content=f"""Position: {current_position}
            Details:
            - Skills: {', '.join(position_details.get('skills', ['']))}
            - Budget/Salary: {position_details.get('budget', 'Competitive')}
            - Timeline: {position_details.get('timeline', 'As soon as possible')}
            - Experience Level: {position_details.get('experience_level', 'Not specified')}
            - Location: {position_details.get('location', 'Not specified')}
            
            {similar_hiring_plans if similar_hiring_plans else ""}
            
            Please create a comprehensive hiring plan and checklist.""")
        ])
        
        plan_chain = plan_prompt | llm
        plan_response = plan_chain.invoke({})
        
        hiring_plan = plan_response.content
    except Exception as e:
        print(f"Error creating hiring plan: {str(e)}")
        # Provide a generic hiring plan if there's an error
        hiring_plan = f"""
                        # Hiring Plan for {current_position}

                        ## 1. Sourcing Candidates
                        - [ ] Post job on major job boards (LinkedIn, Indeed)
                        - [ ] Share position on company website and social media
                        - [ ] Engage recruitment agencies if necessary

                        ## 2. Screening Process
                        - [ ] Review resumes and applications
                        - [ ] Conduct initial phone screenings
                        - [ ] Assess technical skills through preliminary assignments

                        ## 3. Interview Stages
                        - [ ] First round: Technical interview
                        - [ ] Second round: Team fit interview
                        - [ ] Final round: Executive interview

                        ## 4. Decision Making
                        - [ ] Gather feedback from all interviewers
                        - [ ] Make final selection
                        - [ ] Prepare and send offer

                        ## 5. Onboarding
                        - [ ] Prepare equipment and access
                        - [ ] Schedule orientation
                        - [ ] Assign mentor/buddy
                        - [ ] Set up 30/60/90 day plan
                        """
    
    # Store hiring plan in FAISS for future reference
    if RAG_ENABLED and embedding_model is not None:
        faiss_index, document_store = add_to_faiss(
            faiss_index,
            document_store,
            hiring_plan,
            {
                "type": "hiring_plan",
                "position": current_position
            }
        )
        save_faiss_index(faiss_index, document_store)
    
    # Update session state
    session_state["completed_steps"] = session_state.get("completed_steps", []) + ["hiring_plan"]
    session_state["timestamp"] = datetime.datetime.now().isoformat()
    
    # Save hiring plan to file
    plan_dir = os.path.join(DATA_DIR, "hiring_plans")
    os.makedirs(plan_dir, exist_ok=True)
    plan_path = os.path.join(plan_dir, f"{current_position.replace(' ', '_').lower()}_hiring_plan.md")
    try:
        with open(plan_path, 'w') as f:
            f.write(hiring_plan)
        print(f"Saved hiring plan to {plan_path}")
    except Exception as e:
        print(f"Error saving hiring plan to file: {str(e)}")
    
    # Save session state
    if session_id:
        save_session_state(session_id, {
            "current_position": current_position,
            "stage": "market_research",  # Moving to next stage
            "hiring_plans": {
                **state.get("hiring_plans", {}),
                current_position: hiring_plan
            },
            "session_state": session_state,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    # Update the hiring plan in the state
    hiring_plans = state["hiring_plans"].copy()
    hiring_plans[current_position] = hiring_plan
    
    # Update the state
    return {
        **state,
        "hiring_plans": hiring_plans,
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_state": session_state,
        "messages": state["messages"] + [
            AIMessage(content=f"Here's a hiring plan for the {current_position} position:\n\n{hiring_plan}")
        ]
    }


def perform_market_research(state: HRState) -> HRState:
    """Perform market research for the current position to gather industry insights."""
    print("Performing market research...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})
    
    # Build a search query based on position and details
    skills = ", ".join(position_details.get("skills", []))
    experience = position_details.get("experience_level", "")
    location = position_details.get("location", "")
    
    search_query = f"{current_position} salary ranges {experience} {location} {skills}"
    print(f"Search query: {search_query}")
    
    # RAG enhancement: Check if we have similar market research
    existing_research = None
    if RAG_ENABLED and embedding_model is not None:
        query = f"market research {current_position}"
        similar_docs = search_faiss(faiss_index, document_store, query, k=1)
        
        if similar_docs and similar_docs[0]["similarity"] > 0.8:  # High similarity threshold
            top_doc = similar_docs[0]["document"]
            if top_doc["metadata"].get("type") == "market_research":
                existing_research = top_doc["text"]
                print(f"Found existing market research with similarity {similar_docs[0]['similarity']:.2f}")
    
    # Perform the search if no good match in vector store
    try:
        if existing_research:
            print("Using existing market research from FAISS")
            search_results = existing_research
            summary = "Based on our existing knowledge base: " + existing_research.split('\n')[0]
        else:
            # Use the tool's invoke method
            search_results = search_job_market.invoke(search_query)
            
            # Create a summary of the search results
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an HR market researcher.
                Summarize the search results about job market information.
                Focus on salary ranges, required skills, and market demand.
                Keep it concise and actionable."""),
                HumanMessage(content=f"""
                Position: {current_position}
                Search Results: {search_results}
                
                Provide a brief, actionable summary of these search results as they relate to
                hiring for this position, focusing on salary insights, skill requirements, and market demand.
                """)
            ])
            
            summary_chain = summary_prompt | llm
            summary_response = summary_chain.invoke({})
            summary = summary_response.content
            
            # Store in FAISS for future reference
            if RAG_ENABLED and embedding_model is not None:
                market_research_text = f"Market Research for {current_position}:\n{summary}\n\nDetails: {search_results}"
                faiss_index, document_store = add_to_faiss(
                    faiss_index,
                    document_store,
                    market_research_text,
                    {
                        "type": "market_research",
                        "position": current_position,
                        "query": search_query
                    }
                )
                save_faiss_index(faiss_index, document_store)
        
        # Store the search results and summary
        updated_search_results = state.get("search_results", {})
        updated_search_results[current_position] = [
            {"type": "market_research", "query": search_query, "results": search_results, "summary": summary}
        ]
        
        # Update session state
        session_state["completed_steps"] = session_state.get("completed_steps", []) + ["market_research"]
        session_state["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save market research to file
        research_dir = os.path.join(DATA_DIR, "market_research")
        os.makedirs(research_dir, exist_ok=True)
        research_path = os.path.join(research_dir, f"{current_position.replace(' ', '_').lower()}_market_research.md")
        try:
            with open(research_path, 'w') as f:
                f.write(f"# Market Research for {current_position}\n\n{summary}\n\n## Detailed Results\n\n{search_results}")
            print(f"Saved market research to {research_path}")
        except Exception as e:
            print(f"Error saving market research to file: {str(e)}")
        
        # Save session state
        if session_id:
            save_session_state(session_id, {
                "current_position": current_position,
                "stage": "email_templates",  # Moving to next stage
                "search_results": updated_search_results,
                "session_state": session_state,
                "timestamp": datetime.datetime.now().isoformat()
            })
    except Exception as e:
        print(f"Error in market research: {str(e)}")
        summary = f"Unable to complete market research due to an error: {str(e)}"
        updated_search_results = state.get("search_results", {})
    
    # Update the state
    return {
        **state,
        "search_results": updated_search_results,
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_state": session_state,
        "messages": state["messages"] + [
            AIMessage(content=f"I've researched market information for the {current_position} position.\n\n{summary}")
        ]
    }


def create_email_templates(state: HRState) -> HRState:
    """Create common email templates for the hiring process."""
    print("Creating email templates...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})
    
    # Define the email types to generate
    email_types = [
        {"type": "invitation", "recipient": "candidate", "purpose": "interview invitation"},
        {"type": "rejection", "recipient": "candidate", "purpose": "polite rejection"},
        {"type": "offer", "recipient": "candidate", "purpose": "job offer"},
        {"type": "coordination", "recipient": "hiring manager", "purpose": "interview coordination"}
    ]
    
    # Generate email templates
    generated_emails = {}
    email_content_messages = []
    
    for email_config in email_types:
        try:
            # RAG enhancement: Check for similar email templates
            existing_template = None
            if RAG_ENABLED and embedding_model is not None:
                query = f"{email_config['type']} email {current_position}"
                similar_docs = search_faiss(faiss_index, document_store, query, k=1)
                
                if similar_docs and similar_docs[0]["similarity"] > 0.8:  # High similarity threshold
                    top_doc = similar_docs[0]["document"]
                    if (top_doc["metadata"].get("type") == "email_template" and
                        top_doc["metadata"].get("email_type") == email_config["type"]):
                        existing_template = top_doc["text"]
                        print(f"Found existing {email_config['type']} template with similarity {similar_docs[0]['similarity']:.2f}")
            
            if existing_template:
                # Use existing template
                email_content = existing_template
                print(f"Using existing {email_config['type']} template from FAISS")
            else:
                # Prepare custom details
                custom_details = f"""
                Position Details:
                - Skills: {', '.join(position_details.get('skills', []))}
                - Experience: {position_details.get('experience_level', 'Not specified')}
                - Location: {position_details.get('location', 'Not specified')}
                """
                
                # Generate the email - use invoke instead of direct call
                email_content = generate_email_template.invoke({
                    "position": current_position,
                    "email_type": email_config["type"],
                    "recipient_type": email_config["recipient"],
                    "custom_details": custom_details
                })
                
                # Store in FAISS for future reference
                if RAG_ENABLED and embedding_model is not None:
                    faiss_index, document_store = add_to_faiss(
                        faiss_index,
                        document_store,
                        email_content,
                        {
                            "type": "email_template",
                            "position": current_position,
                            "email_type": email_config["type"],
                            "recipient": email_config["recipient"]
                        }
                    )
                    save_faiss_index(faiss_index, document_store)
            
            # Store the generated email
            generated_emails[f"{email_config['type']}_{email_config['recipient']}"] = email_content
            
            # Add this email to our list of messages to display
            email_content_messages.append(
                AIMessage(content=f"### {email_config['type'].title()} Email for {email_config['recipient'].title()}\n\n{email_content}")
            )
            
            # Save email template to file
            template_dir = os.path.join(DATA_DIR, "email_templates", current_position.replace(' ', '_').lower())
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, f"{email_config['type']}_{email_config['recipient']}.txt")
            try:
                with open(template_path, 'w') as f:
                    f.write(email_content)
                print(f"Saved email template to {template_path}")
            except Exception as e:
                print(f"Error saving email template to file: {str(e)}")
            
            # Print the email for debugging
            print(f"Generated {email_config['type']} email for {email_config['recipient']}:\n{email_content}\n")
            
        except Exception as e:
            print(f"Error generating {email_config['type']} email: {str(e)}")
            generated_emails[f"{email_config['type']}_{email_config['recipient']}"] = f"Error generating email: {str(e)}"
            email_content_messages.append(
                AIMessage(content=f"Error generating {email_config['type']} email for {email_config['recipient']}: {str(e)}")
            )
    
    # Update session state
    session_state["completed_steps"] = session_state.get("completed_steps", []) + ["email_templates"]
    session_state["timestamp"] = datetime.datetime.now().isoformat()
    
    # Save session state
    if session_id:
        save_session_state(session_id, {
            "current_position": current_position,
            "stage": "checklists",  # Moving to next stage
            "draft_emails": {
                **state.get("draft_emails", {}),
                current_position: generated_emails
            },
            "session_state": session_state,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    # Update the draft emails in the state
    draft_emails = state.get("draft_emails", {})
    draft_emails[current_position] = generated_emails
    
    # Create an intro message
    intro_message = AIMessage(content=f"I've created the following email templates for the {current_position} position:")
    
    # Update the state with all our messages
    return {
        **state,
        "draft_emails": draft_emails,
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_state": session_state,
        "messages": state["messages"] + [intro_message] + email_content_messages
    }

def build_position_checklists(state: HRState) -> HRState:
    """Build custom checklists for different stages of the hiring process."""
    print("Building custom checklists...")
    
    current_position = state["current_position"]
    faiss_index = state.get("faiss_index")
    document_store = state.get("document_store", [])
    session_id = state.get("session_id")
    session_state = state.get("session_state", {})
    
    # Define checklist types to generate
    checklist_types = [
        {"type": "screening", "description": "Resume screening process"},
        {"type": "interview", "description": "Interview process management"},
        {"type": "onboarding", "description": "New hire onboarding"}
    ]
    
    # Generate checklists
    generated_checklists = {}
    checklist_messages = []
    
    for checklist_config in checklist_types:
        try:
            # RAG enhancement: Look for similar checklists
            existing_checklist = None
            if RAG_ENABLED and embedding_model is not None:
                query = f"{checklist_config['type']} checklist {current_position}"
                similar_docs = search_faiss(faiss_index, document_store, query, k=1)
                
                if similar_docs and similar_docs[0]["similarity"] > 0.8:  # High similarity threshold
                    top_doc = similar_docs[0]["document"]
                    if (top_doc["metadata"].get("type") == "checklist" and
                        top_doc["metadata"].get("checklist_type") == checklist_config["type"]):
                        # Try to extract the JSON checklist from the text
                        try:
                            json_pattern = r'\[.*\]'
                            json_match = re.search(json_pattern, top_doc["text"], re.DOTALL)
                            if json_match:
                                existing_checklist = json.loads(json_match.group(0))
                                print(f"Found existing {checklist_config['type']} checklist with similarity {similar_docs[0]['similarity']:.2f}")
                        except Exception as json_err:
                            print(f"Error extracting checklist JSON: {str(json_err)}")
            
            if existing_checklist:
                # Use existing checklist
                checklist = existing_checklist
                print(f"Using existing {checklist_config['type']} checklist from FAISS")
            else:
                # Generate the checklist - use invoke instead of direct call
                checklist = build_custom_checklist.invoke({
                    "position": current_position,
                    "checklist_type": checklist_config["type"],
                    "specific_items": []  # No specific items required
                })
                
                # Store in FAISS for future reference
                if RAG_ENABLED and embedding_model is not None:
                    checklist_text = f"{checklist_config['type'].capitalize()} Checklist for {current_position}:\n{json.dumps(checklist, indent=2)}"
                    faiss_index, document_store = add_to_faiss(
                        faiss_index,
                        document_store,
                        checklist_text,
                        {
                            "type": "checklist",
                            "position": current_position,
                            "checklist_type": checklist_config["type"]
                        }
                    )
                    
                    # Save the checklist to a file
                    checklist_dir = os.path.join(DATA_DIR, "checklists", current_position.replace(' ', '_').lower())
                    os.makedirs(checklist_dir, exist_ok=True)
                    checklist_path = os.path.join(checklist_dir, f"{checklist_config['type']}_checklist.json")
                    try:
                        with open(checklist_path, 'w') as f:
                            json.dump(checklist, f, indent=2)
                        print(f"Saved {checklist_config['type']} checklist to {checklist_path}")
                    except Exception as e:
                        print(f"Error saving checklist to file: {str(e)}")
            
            # Store the generated checklist
            generated_checklists[checklist_config["type"]] = checklist
            
            # Format the checklist for display
            checklist_display = f"### {checklist_config['type'].title()} Checklist\n\n"
            
            # Format the checklist items
            for i, item in enumerate(checklist, 1):
                category = item.get("category", "General")
                timeline = item.get("timeline", "Anytime")
                assignee = item.get("assignee", "HR")
                task = item.get("item", "Undefined task")
                
                checklist_display += f"{i}. **{task}** (Category: {category}, Timeline: {timeline}, Assignee: {assignee})\n"
            
            # Add this checklist to our list of messages to display
            checklist_messages.append(
                AIMessage(content=checklist_display)
            )
            
            # Print the checklist for debugging
            print(f"Generated {checklist_config['type']} checklist with {len(checklist)} items")
            
        except Exception as e:
            print(f"Error building {checklist_config['type']} checklist: {str(e)}")
            generated_checklists[checklist_config["type"]] = [
                {
                    "item": f"Error creating checklist: {str(e)}",
                    "category": "Error",
                    "timeline": "N/A",
                    "assignee": "HR",
                    "completed": False
                }
            ]
            checklist_messages.append(
                AIMessage(content=f"Error generating {checklist_config['type']} checklist: {str(e)}")
            )
    
    # Save the final FAISS index
    if RAG_ENABLED and embedding_model is not None:
        save_faiss_index(faiss_index, document_store)
    
    # Update session state
    session_state["completed_steps"] = session_state.get("completed_steps", []) + ["checklists"]
    session_state["status"] = "complete"
    session_state["timestamp"] = datetime.datetime.now().isoformat()
    
    # Save session state
    if session_id:
        save_session_state(session_id, {
            "current_position": current_position,
            "stage": "complete",  # Workflow complete
            "checklists": {
                **state.get("checklists", {}),
                current_position: generated_checklists
            },
            "session_state": session_state,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    # Update the checklists in the state
    checklists = state.get("checklists", {})
    checklists[current_position] = generated_checklists
    
    # Create an intro message
    intro_message = AIMessage(content=f"I've created the following detailed checklists for the {current_position} hiring process:")
    
    # Update the state with all our messages
    return {
        **state,
        "checklists": checklists,
        "faiss_index": faiss_index,
        "document_store": document_store,
        "session_state": session_state,
        "messages": state["messages"] + [intro_message] + checklist_messages
    }

# New semantic search tool using FAISS
@tool
def faiss_semantic_search(
    query: str, 
    position: str = None,
    doc_type: str = None,
    k: int = 3
) -> str:
    """
    Perform a semantic search over the HR knowledge base using FAISS.
    
    Args:
        query: The search query
        position: Optional position to filter results by
        doc_type: Optional document type to filter results by
        k: Number of results to return
        
    Returns:
        Formatted search results
    """
    if not RAG_ENABLED or embedding_model is None:
        return "Semantic search is not available without the embedding model."
    
    try:
        # Load FAISS index
        faiss_index, document_store = init_faiss_index()
        
        if len(document_store) == 0:
            return "No documents found in the knowledge base."
        
        # Search for similar documents
        results = search_faiss(faiss_index, document_store, query, k)
        
        if not results:
            return "No relevant documents found."
        
        # Filter by position and doc_type if specified
        if position or doc_type:
            filtered_results = []
            for result in results:
                metadata = result["document"]["metadata"]
                
                # Check if position matches
                position_match = True
                if position:
                    position_match = metadata.get("position", "").lower() == position.lower()
                
                # Check if document type matches
                type_match = True
                if doc_type:
                    type_match = metadata.get("type", "").lower() == doc_type.lower()
                
                # Add to filtered results if both match
                if position_match and type_match:
                    filtered_results.append(result)
            
            results = filtered_results
        
        if not results:
            return f"No documents matched your filters. Position: {position}, Type: {doc_type}"
        
        # Format the results
        formatted_results = f"### Search Results for '{query}'\n\n"
        
        for i, result in enumerate(results, 1):
            doc = result["document"]
            similarity = result["similarity"]
            
            formatted_results += f"**Result {i}** (Similarity: {similarity:.2f})\n"
            formatted_results += f"- Type: {doc['metadata'].get('type', 'Unknown')}\n"
            formatted_results += f"- Position: {doc['metadata'].get('position', 'Unknown')}\n"
            
            # Truncate text for display
            text = doc["text"]
            if len(text) > 300:
                text = text[:300] + "..."
            formatted_results += f"- Content: {text}\n\n"
        
        return formatted_results
    
    except Exception as e:
        return f"Error during semantic search: {str(e)}"

# Tool for loading previous session
@tool
def load_previous_session(session_id: str = None, position: str = None) -> str:
    """
    Load a previous session to continue work.
    
    Args:
        session_id: Specific session ID to load
        position: Position to find most recent session for
        
    Returns:
        Summary of loaded session
    """
    try:
        if not session_id and not position:
            # List available sessions
            sessions = list_available_sessions()
            if not sessions:
                return "No previous sessions found."
                
            result = "Available sessions:\n\n"
            for session in sessions[:5]:  # Show only most recent 5
                result += f"- ID: {session['session_id']}, Position: {session['position']}, Stage: {session['stage']}\n"
            
            return result + "\n\nUse 'load_previous_session(session_id=\"SESSION_ID\")' to load a specific session."
            
        if position and not session_id:
            # Find most recent session for position
            sessions = get_session_history(position)
            if not sessions:
                return f"No sessions found for position: {position}"
                
            # Get most recent
            session_id = sessions[0]["session_id"]
            
        # Load the specified session
        session_data = load_session_state(session_id)
        if not session_data:
            return f"Session {session_id} not found or could not be loaded."
            
        # Return summary
        position = session_data.get("current_position", "Unknown")
        stage = session_data.get("stage", "Unknown")
        timestamp = session_data.get("timestamp", "Unknown")
        session_state = session_data.get("session_state", {})
        
        completed_steps = ", ".join(session_state.get("completed_steps", ["None"]))
        
        return f"""
        Loaded session {session_id} for {position}
        - Current stage: {stage}
        - Last updated: {timestamp}
        - Completed steps: {completed_steps}
        
        You can now continue working with this session.
        """
    
    except Exception as e:
        return f"Error loading session: {str(e)}"

# Export session data
@tool
def export_position_data(position: str, format: str = "json") -> str:
    """
    Export all data for a specific position.
    
    Args:
        position: Position to export data for
        format: Output format (json or markdown)
        
    Returns:
        Path to the exported file
    """
    try:
        # Get all sessions for this position
        sessions = get_session_history(position)
        if not sessions:
            return f"No sessions found for position: {position}"
            
        # Load most recent session
        session_id = sessions[0]["session_id"]
        session_data = load_session_state(session_id)
        
        # Initialize export data
        export_data = {
            "position": position,
            "timestamp": datetime.datetime.now().isoformat(),
            "position_details": session_data.get("position_details", {}).get(position, {}),
            "job_description": session_data.get("job_descriptions", {}).get(position, ""),
            "hiring_plan": session_data.get("hiring_plans", {}).get(position, ""),
            "email_templates": session_data.get("draft_emails", {}).get(position, {}),
            "checklists": session_data.get("checklists", {}).get(position, {})
        }
        
        # Create export directory
        export_dir = os.path.join(DATA_DIR, "exports")
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            # Export as JSON
            export_path = os.path.join(export_dir, f"{position.replace(' ', '_').lower()}_{timestamp}.json")
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            # Export as Markdown
            export_path = os.path.join(export_dir, f"{position.replace(' ', '_').lower()}_{timestamp}.md")
            
            md_content = f"# {position} Hiring Package\n\n"
            md_content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Position details
            md_content += "## Position Details\n\n"
            details = export_data["position_details"]
            if details:
                md_content += f"- **Skills**: {', '.join(details.get('skills', ['None']))}\n"
                md_content += f"- **Budget**: {details.get('budget', 'Not specified')}\n"
                md_content += f"- **Experience**: {details.get('experience_level', 'Not specified')}\n"
                md_content += f"- **Location**: {details.get('location', 'Not specified')}\n"
                md_content += f"- **Timeline**: {details.get('timeline', 'Not specified')}\n\n"
            
            # Job description
            md_content += "## Job Description\n\n"
            md_content += export_data["job_description"] + "\n\n"
            
            # Hiring plan
            md_content += "## Hiring Plan\n\n"
            md_content += export_data["hiring_plan"] + "\n\n"
            
            # Email templates
            md_content += "## Email Templates\n\n"
            for email_type, content in export_data["email_templates"].items():
                md_content += f"### {email_type.replace('_', ' ').title()}\n\n"
                md_content += "```\n" + content + "\n```\n\n"
            
            # Checklists
            md_content += "## Checklists\n\n"
            for checklist_type, items in export_data["checklists"].items():
                md_content += f"### {checklist_type.title()} Checklist\n\n"
                for i, item in enumerate(items, 1):
                    md_content += f"{i}. **{item.get('item')}** - {item.get('category')}, {item.get('timeline')}\n"
                md_content += "\n"
            
            with open(export_path, 'w') as f:
                f.write(md_content)
        
        return f"Exported {position} data to {export_path}"
    except Exception as e:
        return f"Error exporting position data: {str(e)}"

# Keep existing tool definitions
@tool
def search_job_market(query: str) -> str:
    """Search for information about job market, salaries, skills, etc."""
    try:
        print(f"Searching for: {query}")
        
        # Try using Tavily Search API first (requires TAVILY_API_KEY environment variable)
        if os.environ.get("TAVILY_API_KEY"):
            try:
                tavily_search = TavilySearchAPIWrapper()
                # Use results method instead of run
                tavily_results = tavily_search.results(query)
                
                if tavily_results:
                    # Format and return Tavily results
                    if isinstance(tavily_results, str):
                        return tavily_results
                    else:
                        return json.dumps(tavily_results, indent=2)
            except Exception as tavily_error:
                print(f"Tavily search error: {str(tavily_error)}. Falling back to DuckDuckGo.")
        else:
            print("Tavily API key not found. Falling back to DuckDuckGo.")
            
        # Use DuckDuckGo as fallback search
        try:
            # Attempt to import duckduckgo-search
            try:
                import duckduckgo_search
            except ImportError:
                print("Installing duckduckgo-search package...")
                import subprocess
                subprocess.check_call(["pip", "install", "-U", "duckduckgo-search"])
                
            search = DuckDuckGoSearchResults()
            results = search.invoke(query)  # Use invoke instead of run
            
            # Format and limit results
            if isinstance(results, str):
                # Parse the results if they're in string format
                try:
                    parsed_results = json.loads(results)
                    formatted_results = []
                    for result in parsed_results[:3]:  # Limit to 3 results
                        formatted_results.append({
                            "title": result.get("title", "No title"),
                            "snippet": result.get("snippet", "No snippet"),
                            "link": result.get("link", "No link")
                        })
                    return json.dumps(formatted_results, indent=2)
                except:
                    # If parsing fails, return the raw results
                    return results
            else:
                return "No results found"
        except Exception as ddg_error:
            print(f"DuckDuckGo search error: {str(ddg_error)}")
            return "Search engines unavailable. Please check API keys and dependencies."
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return f"Error performing search: {str(e)}"

@tool
def generate_email_template(
    position: str,
    email_type: str,
    recipient_type: str,
    custom_details: str = ""
) -> str:
    """
    Generate an email template for HR purposes.
    
    Args:
        position: The job position relevant to the email
        email_type: Type of email (invitation, rejection, offer, follow-up)
        recipient_type: Who the email is for (candidate, hiring manager, team)
        custom_details: Any specific details to include
        
    Returns:
        An email template as a string
    """
    try:
        print(f"Generating {email_type} email template for {position} position to {recipient_type}")
        
        email_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an HR professional who writes excellent, professional emails.
            Create an email template for the specified purpose.
            Include subject line and body text.
            Keep the tone professional but warm.
            Use appropriate salutations and closings."""),
            HumanMessage(content=f"""
            Create an email template with the following specifications:
            - Position: {position}
            - Email Type: {email_type} (invitation, rejection, offer, or follow-up)
            - Recipient: {recipient_type} (candidate, hiring manager, or team)
            - Custom Details: {custom_details}
            
            Format as:
            Subject: [Subject line]
            
            [Email body with appropriate placeholders]
            
            [Closing]
            """)
        ])
        
        email_chain = email_prompt | llm
        email_result = email_chain.invoke({})
        
        return email_result.content
    except Exception as e:
        print(f"Error generating email template: {str(e)}")
        return f"Error generating email template: {str(e)}"

@tool
def build_custom_checklist(
    position: str,
    checklist_type: str,
    specific_items: List[str] = []
) -> List[Dict[str, Any]]:
    """
    Build a custom checklist for various HR processes.
    
    Args:
        position: The job position the checklist is for
        checklist_type: Type of checklist (screening, interview, onboarding, etc.)
        specific_items: Specific items to include in the checklist
        
    Returns:
        A structured checklist as a list of dictionaries
    """
    try:
        print(f"Building {checklist_type} checklist for {position} position")
        
        # Create a prompt for generating the checklist
        checklist_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an HR operations expert who creates detailed checklists.
            Create a checklist for the specified HR process.
            Return the checklist as a JSON array where each item has:
            - "item": The task description
            - "category": A category for the task
            - "timeline": When this should be done (e.g., "Day 1", "Week 1", "Before interview")
            - "assignee": Who typically handles this (e.g., "HR", "Hiring Manager", "IT")
            - "completed": false (default value)"""),
            HumanMessage(content=f"""
            Create a detailed checklist with the following specifications:
            - Position: {position}
            - Checklist Type: {checklist_type}
            - Required Items: {', '.join(specific_items) if specific_items else 'Standard items for this type'}
            
            The checklist should include at least 5-10 items and cover all important aspects of {checklist_type}.
            """)
        ])
        
        # Set up JSON parser
        parser = JsonOutputParser()
        checklist_chain = checklist_prompt | llm | parser
        
        # Generate the checklist
        try:
            checklist_result = checklist_chain.invoke({})
            
            # Ensure the output is properly formatted
            if isinstance(checklist_result, list):
                return checklist_result
            else:
                # Try to parse if it's a string
                parsed_result = json.loads(checklist_result) if isinstance(checklist_result, str) else []
                return parsed_result if isinstance(parsed_result, list) else []
                
        except Exception as parsing_error:
            print(f"Error parsing checklist: {str(parsing_error)}")
            # Fallback to a basic checklist
            return [
                {
                    "item": f"Set up {checklist_type} process for {position}",
                    "category": "Setup",
                    "timeline": "Start",
                    "assignee": "HR",
                    "completed": False
                },
                {
                    "item": "Prepare necessary documentation",
                    "category": "Documentation",
                    "timeline": "Before process",
                    "assignee": "HR",
                    "completed": False
                },
                {
                    "item": "Schedule necessary meetings",
                    "category": "Scheduling",
                    "timeline": "Before process",
                    "assignee": "HR",
                    "completed": False
                }
            ]
    except Exception as e:
        print(f"Error building checklist: {str(e)}")
        return [
            {
                "item": f"Error creating {checklist_type} checklist: {str(e)}",
                "category": "Error",
                "timeline": "N/A",
                "assignee": "HR",
                "completed": False
            }
        ]


# Step 5: Build the graph
workflow = StateGraph(HRState)

# Add nodes
workflow.add_node("extract_hiring_needs", extract_hiring_needs)
workflow.add_node("ask_clarifying_questions", ask_clarifying_questions)
workflow.add_node("human_input", human_input)
workflow.add_node("process_clarification_responses", process_clarification_responses)
workflow.add_node("create_job_description", create_job_description)
workflow.add_node("create_hiring_plan", create_hiring_plan)
workflow.add_node("perform_market_research", perform_market_research)
workflow.add_node("create_email_templates", create_email_templates)
workflow.add_node("build_position_checklists", build_position_checklists)

# Register tool functions
workflow.add_node("search_job_market_tool", lambda state: {"result": search_job_market})
workflow.add_node("generate_email_template_tool", lambda state: {"result": generate_email_template})
workflow.add_node("build_custom_checklist_tool", lambda state: {"result": build_custom_checklist})
workflow.add_node("faiss_semantic_search_tool", lambda state: {"result": faiss_semantic_search})
workflow.add_node("load_previous_session_tool", lambda state: {"result": load_previous_session})
workflow.add_node("export_position_data_tool", lambda state: {"result": export_position_data})

workflow.add_node("END", lambda state: state)

workflow.add_conditional_edges(
    "process_clarification_responses",
    should_continue_clarifying,
    {
        "continue_clarifying": "ask_clarifying_questions",
        "move_to_job_desc": "create_job_description"
    }
)

# Entry and edges
workflow.set_entry_point("extract_hiring_needs")
workflow.add_edge("extract_hiring_needs", "ask_clarifying_questions")
workflow.add_edge("ask_clarifying_questions", "human_input")
workflow.add_edge("human_input", "process_clarification_responses")
workflow.add_edge("create_job_description", "create_hiring_plan")
workflow.add_edge("create_hiring_plan", "perform_market_research")
workflow.add_edge("perform_market_research", "create_email_templates")
workflow.add_edge("create_email_templates", "build_position_checklists")
workflow.add_edge("build_position_checklists", "END")

# Compile the workflow
hr_helper = workflow.compile()

def run_hr_helper(user_input: str) -> List[Dict[str, Any]]:
    """Run the HR helper with the user's input."""
    print(f"Starting HR Helper with input: {user_input}")
    
    # Check for session management commands
    if user_input.lower().startswith("load session"):
        # Extract session ID if provided
        session_match = re.search(r"load session\s+(.+)$", user_input, re.IGNORECASE)
        session_id = session_match.group(1) if session_match else None
        
        result = load_previous_session(session_id=session_id)
        
        return [
            HumanMessage(content=user_input),
            AIMessage(content=result)
        ]
    
    elif user_input.lower().startswith("list sessions"):
        sessions = list_available_sessions()
        result = "Available sessions:\n\n"
        for session in sessions:
            result += f"- ID: {session['session_id']}, Position: {session['position']}, Stage: {session['stage']}\n"
            
        return [
            HumanMessage(content=user_input),
            AIMessage(content=result)
        ]
    
    elif user_input.lower().startswith("export"):
        # Extract position if provided
        position_match = re.search(r"export\s+(.+)$", user_input, re.IGNORECASE)
        position = position_match.group(1) if position_match else None
        
        if not position:
            return [
                HumanMessage(content=user_input),
                AIMessage(content="Please specify a position to export, e.g., 'export Software Engineer'")
            ]
            
        result = export_position_data(position, format="markdown")
        
        return [
            HumanMessage(content=user_input),
            AIMessage(content=result)
        ]
    
    # Check if it's a semantic search request
    search_pattern = r"^search\s+(?:position:(\w+)\s+)?(?:type:(\w+)\s+)?(.+)$"
    search_match = re.search(search_pattern, user_input, re.IGNORECASE)
    
    if search_match:
        if not RAG_ENABLED or embedding_model is None:
            return [
                HumanMessage(content=user_input),
                AIMessage(content="Semantic search is not available without embeddings. Please install the required dependencies.")
            ]
        
        position = search_match.group(1)
        doc_type = search_match.group(2)
        query = search_match.group(3)
        
        try:
            search_results = faiss_semantic_search(
                query=query,
                position=position,
                doc_type=doc_type,
                k=5
            )
            
            return [
                HumanMessage(content=user_input),
                AIMessage(content=f"I performed a semantic search for '{query}'{f' for position {position}' if position else ''}{f' with document type {doc_type}' if doc_type else ''}.\n\n{search_results}")
            ]
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            return [
                HumanMessage(content=user_input),
                AIMessage(content=f"Error performing semantic search: {str(e)}")
            ]
    
    # Check for resume command from previous session
    resume_match = re.search(r"resume session (\S+)", user_input, re.IGNORECASE)
    if resume_match:
        session_id = resume_match.group(1)
        
        try:
            # Load the session state
            session_data = load_session_state(session_id)
            if not session_data:
                return [
                    HumanMessage(content=user_input),
                    AIMessage(content=f"Session {session_id} not found or could not be loaded.")
                ]
            
            # Create initial state from session data
            messages = session_data.get("messages", [HumanMessage(content=user_input)])
            
            initial_state = {
                "messages": messages,
                "hiring_needs": session_data.get("hiring_needs", []),
                "current_position": session_data.get("current_position", ""),
                "position_details": session_data.get("position_details", {}),
                "job_descriptions": session_data.get("job_descriptions", {}),
                "hiring_plans": session_data.get("hiring_plans", {}),
                "clarification_complete": session_data.get("clarification_complete", False),
                "stage": session_data.get("stage", "clarify"),
                "interaction_count": session_data.get("interaction_count", 0),
                "checklists": session_data.get("checklists", {}),
                "draft_emails": session_data.get("draft_emails", {}),
                "search_results": session_data.get("search_results", {}),
                "faiss_index": None,  # Will be loaded in extract_hiring_needs
                "document_store": [],  # Will be loaded in extract_hiring_needs
                "session_id": session_id,
                "session_state": session_data.get("session_state", {})
            }
            
            # Add a resume message
            additional_message = AIMessage(content=f"Resuming session for {initial_state['current_position']} position from stage: {initial_state['stage']}. Let's continue where we left off.")
            initial_state["messages"].append(additional_message)
            
            # Run the workflow with the resumed state
            try:
                result = hr_helper.invoke(initial_state)
                return result["messages"]
            except Exception as workflow_error:
                print(f"Error running workflow with resumed state: {str(workflow_error)}")
                return [
                    HumanMessage(content=user_input),
                    AIMessage(content=f"Error resuming session: {str(workflow_error)}. Please start a new session.")
                ]
        except Exception as e:
            print(f"Error resuming session: {str(e)}")
            return [
                HumanMessage(content=user_input),
                AIMessage(content=f"Error resuming session {session_id}: {str(e)}")
            ]
    
    # Initialize state for a new session
    session_id = generate_session_id()
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "hiring_needs": [],
        "current_position": "",
        "position_details": {},
        "job_descriptions": {},
        "hiring_plans": {},
        "clarification_complete": False,
        "stage": "clarify",
        "interaction_count": 0,
        "checklists": {},
        "draft_emails": {},
        "search_results": {},
        "faiss_index": None,
        "document_store": [],
        "session_id": session_id,
        "session_state": {}
    }
    
    # Run the graph
    try:
        print("Invoking HR Helper workflow...")
        result = hr_helper.invoke(initial_state)
        print("Workflow completed successfully!")
        
        # Add a completion message
        final_messages = result["messages"]
        
        # Get the position that was worked on
        position = result.get("current_position", "the position")
        session_id = result.get("session_id")
        
        # Save the final FAISS index
        if RAG_ENABLED and embedding_model is not None:
            faiss_index = result.get("faiss_index")
            document_store = result.get("document_store", [])
            if faiss_index is not None:
                save_faiss_index(faiss_index, document_store)
        
        # Add a summary message at the end
        completion_message = AIMessage(content=f"""
        I've completed all steps for hiring a {position}:
        
        1. Gathered position requirements and details
        2. Created a detailed job description
        3. Developed a hiring plan
        4. Performed market research
        5. Generated email templates
        6. Built recruiting process checklists
        
        All this information has been saved to our knowledge base and can be referenced in future hiring processes. 
        Your session ID is: {session_id}
        
        To search this knowledge base, you can use commands like:
        - "search [query]" 
        - "search position:Software Engineer type:job_description [query]"
        
        To continue this session later, use:
        - "resume session {session_id}"
        
        To export all data for this position:
        - "export {position}"
        """)
        
        # Save final state with completion info
        if session_id:
            session_state = result.get("session_state", {})
            session_state["completed"] = True
            session_state["timestamp"] = datetime.datetime.now().isoformat()
            
            save_session_state(session_id, {
                "current_position": position,
                "stage": "complete",
                "session_state": session_state,
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        return final_messages + [completion_message]
    except Exception as e:
        # Get detailed error information
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error running HR helper: {str(e)}")
        print(f"Traceback: {error_traceback}")
        
        # Try to save state even if there was an error
        if session_id:
            try:
                save_session_state(session_id, {
                    "error": str(e),
                    "traceback": error_traceback,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            except Exception as save_error:
                print(f"Error saving error state: {str(save_error)}")
        
        # Simplified direct approach as fallback
        try:
            # Create a basic prompt to handle the request directly
            prompt = f"""You are an HR assistant helping to hire for a position. 
            The user wants to hire someone with this request: "{user_input}"
            What position are they likely hiring for? If unclear, suggest a common position."""
            
            response = llm.invoke(prompt)
            position = response.content.strip()
            
            # Handle edge case where response might not contain a valid position
            if not position or len(position) < 3:
                position = "Software Developer"
                    
            return [
                HumanMessage(content=user_input),
                AIMessage(content=f"I'll help you hire for a {position} position. Please tell me your requirements for this role, including skills needed, experience level, budget, and timeline.")
            ]
        except Exception as fallback_error:
            # Ultimate fallback if even the simplified approach fails
            print(f"Fallback error: {str(fallback_error)}")
            return [
                HumanMessage(content=user_input),
                AIMessage(content="I encountered a technical issue. Please tell me what position you're hiring for and I'll help you create a job description and hiring plan.")
            ]

# Chat interface with RAG search capability
def chat_interface():
    """Simple chat interface for the HR helper."""
    
    print("HR Helper with FAISS RAG and State Persistence - Type 'exit' to quit")
    print("=" * 70)
    print("Special commands:")
    print("- 'search [query]': Search the knowledge base")
    print("- 'search position:Software_Engineer type:job_description [query]': Filtered search")
    print("- 'list sessions': Show all available sessions")
    print("- 'load session': List available sessions")
    print("- 'load session [id]': Load a specific session")
    print("- 'resume session [id]': Resume a previous session")
    print("- 'export [position]': Export all data for a position")
    print("=" * 70)
    
    # Try to load active session if there is one
    active_session_path = os.path.join(STATE_DIR, "active_session.txt")
    active_session_id = None
    
    if os.path.exists(active_session_path):
        try:
            with open(active_session_path, 'r') as f:
                active_session_id = f.read().strip()
            
            active_session = load_session_state(active_session_id)
            if active_session:
                position = active_session.get("current_position", "unknown position")
                print(f"Found active session {active_session_id} for {position}")
                
                # Ask if user wants to resume
                resume = input(f"Would you like to resume your session for {position}? (y/n): ").lower()
                if resume.startswith('y'):
                    # Run resume command
                    messages = run_hr_helper(f"resume session {active_session_id}")
                    conversation = messages
                    
                    # Print AI responses
                    for message in conversation:
                        if isinstance(message, AIMessage):
                            print(f"HR Helper: {message.content}")
                    
                    # Continue with resumed session below
                else:
                    # Start new session
                    active_session_id = None
                    if os.path.exists(active_session_path):
                        os.remove(active_session_path)
        except Exception as e:
            print(f"Error loading active session: {str(e)}")
            active_session_id = None
    
    # Start chat loop
    conversation = []
    if active_session_id and 'conversation' in locals():
        # If we resumed an active session above
        pass
    else:
        # Start new conversation
        user_input = input("You: ")
        
        while user_input.lower() != "exit":
            if not conversation:
                # First message, start the workflow
                messages = run_hr_helper(user_input)
                conversation = messages
                
                # Get session ID if available
                for msg in messages:
                    if isinstance(msg, AIMessage) and "Your session ID is:" in msg.content:
                        session_match = re.search(r"Your session ID is:\s*(\S+)", msg.content)
                        if session_match:
                            active_session_id = session_match.group(1)
                            # Save as active session
                            try:
                                with open(active_session_path, 'w') as f:
                                    f.write(active_session_id)
                            except Exception as e:
                                print(f"Warning: Could not save active session: {str(e)}")
            else:
                # Continue the conversation
                conversation.append(HumanMessage(content=user_input))
                
                # Check for special commands
                if user_input.lower().startswith(("search ", "list sessions", "load session", "resume session", "export ")):
                    # Handle special commands
                    special_result = run_hr_helper(user_input)
                    conversation.extend(special_result[1:])  # Add the AI response
                else:
                    # Create a new state with the updated conversation
                    state = {
                        "messages": conversation,
                        "hiring_needs": [],  # These will be populated by the workflow
                        "current_position": "",
                        "position_details": {},
                        "job_descriptions": {},
                        "hiring_plans": {},
                        "clarification_complete": False,
                        "stage": "clarify",
                        "interaction_count": 0,
                        "checklists": {},
                        "draft_emails": {},
                        "search_results": {},
                        "faiss_index": None,
                        "document_store": [],
                        "session_id": active_session_id,
                        "session_state": {}
                    }
                    
                    try:
                        # Run the workflow with the updated state
                        result = hr_helper.invoke(state)
                        conversation = result["messages"]
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        conversation.append(AIMessage(content="I encountered an error. Can you please rephrase your request?"))
            
            # Print AI responses
            for message in conversation:
                if isinstance(message, AIMessage) and message.content not in [m.content for m in conversation[:conversation.index(message)]]:
                    print(f"HR Helper: {message.content}")
            
            # Get next user input
            user_input = input("You: ")
        
        # Clear active session on exit
        if os.path.exists(active_session_path):
            try:
                os.remove(active_session_path)
            except Exception as e:
                print(f"Warning: Could not clear active session: {str(e)}")
    
    print("Thank you for using HR Helper!")

# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HR Helper with FAISS RAG and State Persistence")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--setup", "-s", action="store_true", help="Setup and initialize FAISS with sample data")
    parser.add_argument("--list", "-l", action="store_true", help="List all saved sessions")
    parser.add_argument("--resume", "-r", help="Resume a specific session ID")
    parser.add_argument("--export", "-e", help="Export data for a position")
    parser.add_argument("query", nargs="?", help="Initial query or search request")
    
    args = parser.parse_args()
    
    # Display version info
    print("Starting HR Helper application with FAISS RAG and State Persistence")
    
    # List sessions
    if args.list:
        sessions = list_available_sessions()
        if not sessions:
            print("No saved sessions found.")
        else:
            print("Available sessions:")
            for session in sessions:
                print(f"- ID: {session['session_id']}, Position: {session['position']}, Stage: {session['stage']}")
        sys.exit(0)
    
    # Resume session
    if args.resume:
        messages = run_hr_helper(f"resume session {args.resume}")
        
        # Print only AI messages
        for message in messages:
            if isinstance(message, AIMessage):
                print(f"HR Helper: {message.content}")
        sys.exit(0)
    
    # Export position
    if args.export:
        result = export_position_data(args.export, format="markdown")
        print(result)
        sys.exit(0)
    
    # Setup mode
    if args.setup:
        print("Setting up FAISS index with sample data...")
        
        # Initialize FAISS
        faiss_index, document_store = init_faiss_index()
        
        # Add sample documents
        sample_documents = [
            {
                "text": "Software Engineer job requires expertise in Python, JavaScript, and cloud technologies.",
                "metadata": {"type": "job_description", "position": "Software Engineer"}
            },
            {
                "text": "Data Scientist position needs machine learning, statistics, and data visualization skills.",
                "metadata": {"type": "job_description", "position": "Data Scientist"}
            },
            {
                "text": "UX Designer role focuses on user research, wireframing, and prototyping.",
                "metadata": {"type": "job_description", "position": "UX Designer"}
            },
            {
                "text": """Subject: Interview Invitation - Software Engineer Position

Dear [Candidate Name],

We were impressed by your application for the Software Engineer position and would like to invite you for an interview.

Best regards,
HR Team""",
                "metadata": {"type": "email_template", "position": "Software Engineer", "email_type": "invitation", "recipient": "candidate"}
            }
        ]
        
        # Add documents to FAISS
        for doc in sample_documents:
            faiss_index, document_store = add_to_faiss(
                faiss_index,
                document_store,
                doc["text"],
                doc["metadata"]
            )
        
        # Save FAISS index
        save_faiss_index(faiss_index, document_store)
        
        print(f"FAISS index initialized with {len(document_store)} sample documents")
        sys.exit(0)
    
    # Run in interactive mode
    if args.interactive:
        chat_interface()
    # Run with query from command line
    elif args.query:
        messages = run_hr_helper(args.query)
        
        # Print only AI messages
        for message in messages:
            if isinstance(message, AIMessage):
                print(f"HR Helper: {message.content}")
    else:
        # Start interactive chat
        chat_interface()