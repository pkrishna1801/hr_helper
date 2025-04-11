import streamlit as st
import os
import json
import re
import datetime
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Import LangChain components for LLM integration
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
try:
    from langchain_openai import ChatOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    st.sidebar.warning("LangChain OpenAI not available. Install with: pip install langchain-openai")

# Optional imports - will use if available
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RAG_ENABLED = True
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    except Exception as e:
        st.sidebar.warning(f"Could not load embedding model: {str(e)}")
        embedding_model = None
        EMBEDDING_DIM = 384  # Default for MiniLM-L6-v2
        RAG_ENABLED = False
except ImportError:
    RAG_ENABLED = False
    embedding_model = None
    EMBEDDING_DIM = 384

# Setup directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vectors")
STATE_DIR = os.path.join(DATA_DIR, "state")
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "job_descriptions"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "hiring_plans"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "email_templates"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "checklists"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "exports"), exist_ok=True)

# State Management Functions
def generate_session_id() -> str:
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
        
        return True
    except Exception as e:
        st.error(f"Error saving session state: {str(e)}")
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
        
        return state_data
    except Exception as e:
        st.error(f"Error loading session state: {str(e)}")
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
                st.error(f"Error reading session {session_id}: {str(e)}")
    
    return sorted(sessions, key=lambda s: s.get('time', ''), reverse=True)

def get_session_history(position: str = None) -> List[Dict[str, Any]]:
    """Get history of sessions for a specific position or all sessions."""
    all_sessions = list_available_sessions()
    
    if position:
        return [s for s in all_sessions if s.get('position', '').lower() == position.lower()]
    else:
        return all_sessions

# RAG Utility Functions
def init_faiss_index() -> Tuple[Union[Any, None], List[Dict[str, Any]]]:
    """Initialize or load FAISS index."""
    if not RAG_ENABLED:
        return None, []
        
    try:
        index_path = os.path.join(VECTOR_DIR, "hr_index.faiss")
        docs_path = os.path.join(VECTOR_DIR, "hr_docs.json")
        
        # Create new index if none exists
        if not os.path.exists(index_path):
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            documents = []
            return index, documents
        
        # Load existing index and documents
        index = faiss.read_index(index_path)
        
        with open(docs_path, "r") as f:
            documents = json.load(f)
        
        return index, documents
    except Exception as e:
        st.error(f"Error initializing FAISS index: {str(e)}")
        # Return empty index and documents as fallback
        if RAG_ENABLED:
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            return index, []
        else:
            return None, []

def save_faiss_index(index, documents) -> bool:
    """Save FAISS index and documents to disk."""
    if not RAG_ENABLED or index is None:
        return False
        
    try:
        os.makedirs(VECTOR_DIR, exist_ok=True)
        index_path = os.path.join(VECTOR_DIR, "hr_index.faiss")
        docs_path = os.path.join(VECTOR_DIR, "hr_docs.json")
        
        # Save the FAISS index
        faiss.write_index(index, index_path)
        
        # Save the documents
        with open(docs_path, "w") as f:
            json.dump(documents, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving FAISS index: {str(e)}")
        return False

def add_to_faiss(text: str, metadata: Dict[str, Any]) -> None:
    """Add a document to the FAISS index."""
    if not RAG_ENABLED or embedding_model is None:
        return
    
    try:
        # Initialize or load FAISS
        index, documents = init_faiss_index()
        if index is None:
            return
            
        # Add timestamp to metadata
        metadata = {
            **metadata,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Create embedding
        embedding = embedding_model.encode([text])[0]
        vector = embedding.reshape(1, -1).astype('float32')
        
        # Add to index
        index.add(vector)
        
        # Add to documents store
        documents.append({
            "text": text,
            "metadata": metadata
        })
        
        # Save updated index
        save_faiss_index(index, documents)
        
    except Exception as e:
        st.error(f"Error adding to FAISS: {str(e)}")

def search_faiss(query: str, position: str = None, doc_type: str = None, k: int = 3) -> List[Dict[str, Any]]:
    """Search FAISS index for similar documents."""
    if not RAG_ENABLED or embedding_model is None:
        return []
    
    try:
        # Load FAISS index
        index, documents = init_faiss_index()
        if index is None or len(documents) == 0:
            return []
            
        # Create embedding for query
        embedding = embedding_model.encode([query])[0]
        vector = embedding.reshape(1, -1).astype('float32')
        
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
                doc = documents[idx]
                metadata = doc.get("metadata", {})
                
                # Filter by position and doc_type if specified
                position_match = True
                if position:
                    position_match = metadata.get("position", "").lower() == position.lower()
                
                type_match = True
                if doc_type:
                    type_match = metadata.get("type", "").lower() == doc_type.lower()
                
                # Add to results if it matches filters
                if position_match and type_match:
                    results.append({
                        "document": doc,
                        "distance": float(distances[0][i]),
                        "similarity": 1.0 / (1.0 + float(distances[0][i]))
                    })
        
        return results
    except Exception as e:
        st.error(f"Error searching FAISS: {str(e)}")
        return []

def extract_position_with_llm(user_input: str) -> str:
    """Extract position using LLM if available, otherwise use regex."""
    if LLM_AVAILABLE:
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
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
            
            system_message = SystemMessage(content="You are an expert at extracting hiring information from text.")
            human_message = HumanMessage(content=extraction_prompt)
            
            response = llm.invoke([system_message, human_message])
            
            # Try to parse JSON response
            try:
                import json
                extracted_info = json.loads(response.content)
                position = extracted_info.get("position", "Software Developer")
                return position
            except:
                # Fallback to regex if JSON parsing fails
                match = re.search(r"\"position\":\s*\"([^\"]+)\"", response.content)
                if match:
                    return match.group(1)
                else:
                    return "Software Developer"
                    
        except Exception as e:
            st.warning(f"LLM extraction failed: {str(e)}. Using fallback method.")
    
    # Fallback using regex patterns
    common_positions = [
        "Software Engineer", "Data Scientist", "Product Manager", "UX Designer", 
        "Marketing Manager", "Sales Representative", "HR Manager", "Financial Analyst",
        "Project Manager", "Business Analyst", "Front-end Developer", "Back-end Developer",
        "Full Stack Developer", "DevOps Engineer", "QA Engineer", "Engineering Manager",
        "Content Writer", "Graphic Designer", "Customer Success Manager", "Operations Manager"
    ]
    
    # Try to find position using regex
    match = re.search(r"hire (?:a|an)\s+([A-Za-z\s]+(?:Developer|Engineer|Designer|Manager|Analyst|Scientist|Representative|Writer))", user_input)
    if match:
        return match.group(1).strip()
    
    # Look for position mentioned with keywords
    for position in common_positions:
        if position.lower() in user_input.lower():
            return position
    
    # Default
    return "Software Developer"

def export_position_data(position: str, format: str = "markdown") -> str:
    """Export all data for a specific position."""
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
        
        return export_path
    except Exception as e:
        st.error(f"Error exporting position data: {str(e)}")
        return f"Error: {str(e)}"

# LLM Integration Functions
def get_llm():
    """Get LLM instance if available."""
    if not LLM_AVAILABLE:
        return None
    
    try:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def generate_with_llm(prompt, system_prompt="You are an HR assistant helping with hiring processes."):
    """Generate text using LLM."""
    llm = get_llm()
    if not llm:
        return "LLM not available. Please install langchain-openai and set your API key."
    
    try:
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=prompt)
        
        response = llm.invoke([system_message, human_message])
        return response.content
    except Exception as e:
        st.error(f"Error generating with LLM: {str(e)}")
        return f"Error generating response: {str(e)}"

def generate_job_description(position, details):
    """Generate a job description using LLM if available."""
    skills = ', '.join(details.get('skills', ['relevant technical skills']))
    experience = details.get('experience_level', 'mid-level')
    location = details.get('location', 'remote')
    budget = details.get('budget', 'competitive')
    
    if LLM_AVAILABLE:
        prompt = f"""
        Create a detailed job description for a {position} position with the following details:
        - Skills: {skills}
        - Experience level: {experience}
        - Location: {location}
        - Budget/Salary: {budget}
        
        Format the job description in Markdown with the following sections:
        - About the role
        - Responsibilities
        - Requirements
        - Nice-to-have skills
        - Benefits
        - Application process
        """
        
        system_prompt = """You are an experienced HR professional and technical recruiter. 
        Create a compelling and detailed job description for the specified position.
        Make the description professional, engaging, and formatted in Markdown."""
        
        return generate_with_llm(prompt, system_prompt)
    else:
        # Fallback without LLM
        return f"""
        # {position}

        ## About the Role
        We are seeking a talented {position} to join our team. This {experience} role offers an opportunity to work on exciting projects and make a significant impact.

        ## Responsibilities
        - Collaborate with team members to achieve project goals
        - Apply technical expertise to solve complex problems
        - Stay updated with industry trends and technologies

        ## Requirements
        - Experience with {skills}
        - Strong problem-solving and analytical skills
        - Excellent communication and teamwork abilities

        ## Benefits
        - {budget} salary
        - Professional development opportunities
        - {location} work arrangement

        ## How to Apply
        Please submit your resume and a brief cover letter outlining your qualifications.
        """

def generate_hiring_plan(position, details):
    """Generate a hiring plan using LLM if available."""
    timeline = details.get('timeline', 'As soon as possible')
    skills = ', '.join(details.get('skills', ['relevant technical skills']))
    
    if LLM_AVAILABLE:
        prompt = f"""
        Create a detailed hiring plan/checklist for a {position} position with the following details:
        - Skills: {skills}
        - Timeline: {timeline}
        
        Format the hiring plan as a Markdown checklist with these sections:
        - Sourcing candidates
        - Screening process
        - Interview stages
        - Assessment methods
        - Decision timeline
        - Onboarding steps
        """
        
        system_prompt = """You are an experienced HR professional. 
        Create a detailed hiring plan/checklist for the specified position.
        Format the plan as a structured Markdown checklist with main sections and sub-items."""
        
        return generate_with_llm(prompt, system_prompt)
    else:
        # Fallback without LLM
        return f"""
        # Hiring Plan for {position}

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
        
        ## Timeline
        - Target completion: {timeline}
        """

def generate_email_template(position, email_type, recipient):
    """Generate an email template using LLM if available."""
    if LLM_AVAILABLE:
        prompt = f"""
        Create an email template for a {position} position with the following specifications:
        - Email Type: {email_type} (invitation, rejection, offer, or follow-up)
        - Recipient: {recipient} (candidate, hiring manager, or team)
        
        Format as:
        Subject: [Subject line]
        
        [Email body with appropriate placeholders]
        
        [Closing]
        """
        
        system_prompt = """You are an HR professional who writes excellent, professional emails.
        Create an email template for the specified purpose.
        Include subject line and body text.
        Keep the tone professional but warm.
        Use appropriate salutations and closings."""
        
        return generate_with_llm(prompt, system_prompt)
    else:
        # Fallback without LLM
        if email_type == "invitation":
            return f"""Subject: Interview Invitation - {position} Position

Dear [Candidate Name],

We were impressed by your application for the {position} position and would like to invite you for an interview.

The interview will take place on [Date] at [Time] [Location/Virtual Platform].

Please confirm your availability by replying to this email.

Best regards,
[Your Name]
HR Department"""
        
        elif email_type == "rejection":
            return f"""Subject: Regarding Your Application for {position}

Dear [Candidate Name],

Thank you for your interest in the {position} position and for taking the time to go through our interview process.

After careful consideration, we regret to inform you that we have decided to proceed with another candidate whose experience better aligns with our current needs.

We appreciate your interest in our company and wish you success in your job search.

Best regards,
[Your Name]
HR Department"""
        
        elif email_type == "offer":
            return f"""Subject: Job Offer - {position}

Dear [Candidate Name],

We are pleased to offer you the position of {position} at [Company Name].

The details of your offer are as follows:
- Start Date: [Start Date]
- Salary: [Salary]
- Benefits: [Benefits Summary]

Please review the attached formal offer letter for complete details.

We would appreciate your response by [Response Deadline].

Best regards,
[Your Name]
HR Department"""
        
        else:  # coordination
            return f"""Subject: {position} Interview Coordination

Hello [Hiring Manager],

I've scheduled interviews for the {position} role for the following candidates:

1. [Candidate Name] - [Date/Time]
2. [Candidate Name] - [Date/Time]

Please let me know if these times work for you or if adjustments are needed.

Best regards,
[Your Name]
HR Department"""

def generate_checklist(position, checklist_type):
    """Generate a checklist using LLM if available."""
    if LLM_AVAILABLE:
        prompt = f"""
        Create a detailed checklist for a {position} position with the following specifications:
        - Checklist Type: {checklist_type} (screening, interview, onboarding)
        
        Return the checklist as a JSON array where each item has:
        - "item": The task description
        - "category": A category for the task
        - "timeline": When this should be done
        - "assignee": Who typically handles this
        - "completed": false (default value)
        
        The checklist should include at least 7-10 items.
        """
        
        system_prompt = """You are an HR operations expert who creates detailed checklists.
        Create a checklist for the specified HR process.
        Return the checklist as a JSON array with the specified fields."""
        
        response = generate_with_llm(prompt, system_prompt)
        
        # Try to parse JSON response
        try:
            import json
            pattern = r'\[\s*\{.*\}\s*\]'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0)
                checklist_items = json.loads(json_str)
                return checklist_items
        except:
            # If JSON parsing fails, return default
            pass
    
    # Fallback checklist
    if checklist_type == "screening":
        return [
            {"item": "Review application for completeness", "category": "Initial Review", "timeline": "Day 1", "assignee": "HR", "completed": False},
            {"item": "Check candidate qualifications against job requirements", "category": "Initial Review", "timeline": "Day 1", "assignee": "HR", "completed": False},
            {"item": "Screen for required certifications/education", "category": "Verification", "timeline": "Day 1", "assignee": "HR", "completed": False},
            {"item": "Schedule initial phone screening", "category": "Coordination", "timeline": "Day 2", "assignee": "HR", "completed": False},
            {"item": "Conduct phone screening", "category": "Assessment", "timeline": "Day 3-5", "assignee": "HR", "completed": False},
            {"item": "Document screening results", "category": "Documentation", "timeline": "Same day as screening", "assignee": "HR", "completed": False},
            {"item": "Make decision for next round", "category": "Decision", "timeline": "Day 5", "assignee": "HR/Hiring Manager", "completed": False}
        ]
    elif checklist_type == "interview":
        return [
            {"item": "Create interview schedule", "category": "Planning", "timeline": "Week 1", "assignee": "HR", "completed": False},
            {"item": "Prepare interview questions", "category": "Preparation", "timeline": "Week 1", "assignee": "Hiring Manager", "completed": False},
            {"item": "Send interview invitations", "category": "Coordination", "timeline": "Week 1", "assignee": "HR", "completed": False},
            {"item": "Prepare interview panel", "category": "Preparation", "timeline": "Week 1", "assignee": "Hiring Manager", "completed": False},
            {"item": "Conduct technical interviews", "category": "Assessment", "timeline": "Week 2", "assignee": "Technical Team", "completed": False},
            {"item": "Conduct culture fit interviews", "category": "Assessment", "timeline": "Week 2", "assignee": "HR/Team Lead", "completed": False},
            {"item": "Collect feedback from interviewers", "category": "Assessment", "timeline": "Week 2", "assignee": "HR", "completed": False},
            {"item": "Make hiring decision", "category": "Decision", "timeline": "Week 3", "assignee": "Hiring Manager", "completed": False}
        ]
    else:  # onboarding
        return [
            {"item": "Send welcome email", "category": "Welcome", "timeline": "1 week before start", "assignee": "HR", "completed": False},
            {"item": "Prepare workstation/equipment", "category": "IT Setup", "timeline": "1 week before start", "assignee": "IT", "completed": False},
            {"item": "Set up accounts and access", "category": "IT Setup", "timeline": "1 week before start", "assignee": "IT", "completed": False},
            {"item": "Prepare first-day schedule", "category": "Planning", "timeline": "3 days before start", "assignee": "HR", "completed": False},
            {"item": "Assign onboarding buddy", "category": "Support", "timeline": "1 week before start", "assignee": "Team Lead", "completed": False},
            {"item": "First day orientation", "category": "Orientation", "timeline": "Day 1", "assignee": "HR", "completed": False},
            {"item": "Team introduction", "category": "Integration", "timeline": "Day 1", "assignee": "Team Lead", "completed": False},
            {"item": "Set up 30/60/90 day plan", "category": "Planning", "timeline": "Week 1", "assignee": "Hiring Manager", "completed": False},
            {"item": "First week check-in", "category": "Support", "timeline": "Day 5", "assignee": "HR", "completed": False},
            {"item": "30-day review", "category": "Review", "timeline": "Day 30", "assignee": "Hiring Manager", "completed": False}
        ]

# Function to store conversation in session
def add_to_conversation(role, message):
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    st.session_state.conversation.append({"role": role, "message": message, "time": datetime.datetime.now().isoformat()})

# Main application
def main():
    st.set_page_config(
        page_title="HR Helper",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None
    if "current_position" not in st.session_state:
        st.session_state.current_position = ""
    if "position_details" not in st.session_state:
        st.session_state.position_details = {}
    if "job_descriptions" not in st.session_state:
        st.session_state.job_descriptions = {}
    if "hiring_plans" not in st.session_state:
        st.session_state.hiring_plans = {}
    if "draft_emails" not in st.session_state:
        st.session_state.draft_emails = {}
    if "checklists" not in st.session_state:
        st.session_state.checklists = {}
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Dashboard"
    
    # Application header
    st.title("HR Helper")
    st.markdown("### Streamlining your hiring process")
    
    # Sidebar for navigation and session management
    with st.sidebar:
        st.header("Navigation")
        selected_tab = st.radio(
            "Go to:",
            ["Dashboard", "Position Details", "Job Description", "Hiring Plan", 
             "Email Templates", "Checklists", "Knowledge Base", "Export"]
        )
        st.session_state.active_tab = selected_tab
        
        st.markdown("---")
        
        # Session management
        st.header("Session Management")
        
        # List available sessions
        sessions = list_available_sessions()
        if sessions:
            session_options = ["New Session"] + [f"{s['position']} ({s['session_id']})" for s in sessions]
            selected_session = st.selectbox("Session", session_options)
            
            if selected_session != "New Session" and st.button("Load Selected Session"):
                # Extract session ID from selection
                session_id = re.search(r"\((.*?)\)$", selected_session).group(1)
                session_data = load_session_state(session_id)
                
                if session_data:
                    st.session_state.active_session_id = session_id
                    st.session_state.current_position = session_data.get("current_position", "")
                    st.session_state.position_details = session_data.get("position_details", {})
                    st.session_state.job_descriptions = session_data.get("job_descriptions", {})
                    st.session_state.hiring_plans = session_data.get("hiring_plans", {})
                    st.session_state.draft_emails = session_data.get("draft_emails", {})
                    st.session_state.checklists = session_data.get("checklists", {})
                    
                    # Load conversation if available
                    if "messages" in session_data:
                        st.session_state.conversation = []
                        for msg in session_data.get("messages", []):
                            if hasattr(msg, "content"):
                                role = "assistant" if msg.type == "ai" else "user"
                                add_to_conversation(role, msg.content)
                    
                    st.success(f"Loaded session for {st.session_state.current_position}")
                    st.rerun()
            
        if st.session_state.active_session_id:
            if st.button("Start New Session"):
                st.session_state.active_session_id = None
                st.session_state.current_position = ""
                st.session_state.position_details = {}
                st.session_state.job_descriptions = {}
                st.session_state.hiring_plans = {}
                st.session_state.draft_emails = {}
                st.session_state.checklists = {}
                st.session_state.conversation = []
                st.rerun()
        
        # Show active session info
        if st.session_state.active_session_id:
            st.markdown("---")
            st.markdown(f"**Active Session**: {st.session_state.active_session_id}")
            st.markdown(f"**Position**: {st.session_state.current_position}")
        
        # API key input for OpenAI (if needed)
        if not os.environ.get("OPENAI_API_KEY") and LLM_AVAILABLE:
            st.markdown("---")
            st.subheader("API Configuration")
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key set!")
    
    # Dashboard tab
    if st.session_state.active_tab == "Dashboard":
        st.header("Dashboard")
        
        if not st.session_state.active_session_id:
            # New session setup
            st.subheader("Start a New Hiring Process")
            user_input = st.text_area("What position are you hiring for?", 
                                      placeholder="Example: We need to hire a Senior Software Engineer with Python and AWS experience. Budget is around $120k-$150k. Remote work is possible.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Hiring Process", type="primary"):
                    if user_input:
                        # Extract position from input text
                        position = extract_position_with_llm(user_input)
                        
                        # Generate a new session ID
                        session_id = generate_session_id()
                        st.session_state.active_session_id = session_id
                        st.session_state.current_position = position
                        
                        # Initialize position details - will be populated through clarification
                        st.session_state.position_details = {
                            position: {
                                "skills": [],
                                "budget": None,
                                "timeline": None,
                                "experience_level": None,
                                "location": None
                            }
                        }
                        
                        # Add to conversation
                        add_to_conversation("user", user_input)
                        add_to_conversation("assistant", f"I'll help you hire for the {position} position. Let's start by gathering some details.")
                        
                        # Initialize storage
                        st.session_state.job_descriptions = {}
                        st.session_state.hiring_plans = {}
                        
                        # Save initial session
                        session_state = {
                            "start_time": datetime.datetime.now().isoformat(),
                            "position": position,
                            "completed_steps": ["extract_hiring_needs"],
                            "position_history": [position],
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        save_session_state(session_id, {
                            "current_position": position,
                            "stage": "clarify",
                            "session_state": session_state,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                        st.success(f"Created new session for {position}")
                        st.rerun()
                    else:
                        st.error("Please enter a position to continue")
            
            with col2:
                # Display recent sessions
                if sessions:
                    st.markdown("**Recent Sessions**")
                    for i, session in enumerate(sessions[:5]):
                        st.markdown(f"- {session['position']} ({session['time'][:10]})")
        else:
            # Display active session info and progress
            st.subheader(f"Hiring Process for {st.session_state.current_position}")
            
            # Display progress with metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate completion metrics
            details_completion = len([v for v in st.session_state.position_details.get(st.session_state.current_position, {}).values() if v]) / 5 * 100
            has_job_desc = st.session_state.current_position in st.session_state.job_descriptions
            has_hiring_plan = st.session_state.current_position in st.session_state.hiring_plans
            has_emails = st.session_state.current_position in st.session_state.draft_emails
            has_checklists = st.session_state.current_position in st.session_state.checklists
            
            # Count complete steps
            complete_steps = sum([
                details_completion >= 60,  # Position details at least 60% complete
                has_job_desc,
                has_hiring_plan,
                has_emails,
                has_checklists
            ])
            
            # Calculate overall completion percentage
            overall_completion = complete_steps / 5 * 100
            
            with col1:
                st.metric("Position Details", f"{details_completion:.0f}%")
            with col2:
                st.metric("Job Description", "Complete" if has_job_desc else "Pending")
            with col3:
                st.metric("Hiring Plan", "Complete" if has_hiring_plan else "Pending")
            with col4:
                st.metric("Overall Progress", f"{overall_completion:.0f}%")
            
            # Progress bar
            st.progress(overall_completion / 100)
            
            # Next steps
            st.subheader("Next Steps")
            if details_completion < 60:
                st.info("➡️ Complete position details to continue")
            elif not has_job_desc:
                st.info("➡️ Create job description")
            elif not has_hiring_plan:
                st.info("➡️ Create hiring plan")
            elif not has_emails:
                st.info("➡️ Generate email templates")
            elif not has_checklists:
                st.info("➡️ Build hiring checklists")
            else:
                st.success("✅ All steps completed! You can export the hiring package.")
            
            # Quick actions
            st.subheader("Quick Actions")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("Create Job Description"):
                    st.session_state.active_tab = "Job Description"
                    st.rerun()
            
            with action_col2:
                if st.button("Create Hiring Plan"):
                    st.session_state.active_tab = "Hiring Plan"
                    st.rerun()
            
            with action_col3:
                if st.button("Generate Email Templates"):
                    st.session_state.active_tab = "Email Templates"
                    st.rerun()
            
            # Chat interface for the current session
            st.subheader("HR Assistant Chat")
            user_message = st.text_input("Ask a question or provide information about the position...")
            
            if st.button("Send", key="chat_send") and user_message:
                # Add user message to conversation
                add_to_conversation("user", user_message)
                
                # Generate assistant response
                if LLM_AVAILABLE:
                    position = st.session_state.current_position
                    details = st.session_state.position_details.get(position, {})
                    
                    prompt = f"""
                    I'm helping to hire for a {position} position. The user says: "{user_message}"
                    
                    Current position details:
                    - Skills: {', '.join(details.get('skills', []))}
                    - Experience level: {details.get('experience_level', 'Not specified')}
                    - Location: {details.get('location', 'Not specified')}
                    - Budget: {details.get('budget', 'Not specified')}
                    - Timeline: {details.get('timeline', 'Not specified')}
                    
                    Respond helpfully with HR expertise. Ask for more details if needed.
                    """
                    
                    response = generate_with_llm(prompt)
                else:
                    response = f"I'll help you with your query about the {st.session_state.current_position} position. To use the AI assistant functionality, please set up your OpenAI API key."
                
                # Add assistant response to conversation
                add_to_conversation("assistant", response)
                
                # Save updated conversation to session
                if st.session_state.active_session_id:
                    save_session_state(st.session_state.active_session_id, {
                        "current_position": st.session_state.current_position,
                        "conversation": st.session_state.conversation,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                st.rerun()
            
            # Display conversation
            st.subheader("Conversation History")
            for item in st.session_state.conversation:
                if item["role"] == "user":
                    st.markdown(f"""
                    <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>You:</strong> {item['message']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>HR Helper:</strong> {item['message']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Position Details tab
    elif st.session_state.active_tab == "Position Details":
        st.header("Position Details")
        
        if not st.session_state.active_session_id:
            st.warning("Please start a new session from the Dashboard.")
            return
        
        position = st.session_state.current_position
        details = st.session_state.position_details.get(position, {})
        
        # Display form for position details
        with st.form("position_details_form"):
            st.subheader(f"Details for {position}")
            
            # Skills - multiselect with custom input
            default_skills = details.get("skills", [])
            skill_options = ["Python", "JavaScript", "React", "Data Analysis", "Project Management", 
                           "Leadership", "Communication", "Problem-solving", "Java", "C++", 
                           "AWS", "Azure", "DevOps", "UI/UX Design", "Machine Learning"] + default_skills
            
            skills = st.multiselect(
                "Required Skills", 
                options=sorted(list(set(skill_options))),
                default=default_skills
            )
            
            # Additional skills input
            additional_skills = st.text_input("Add Other Skills (comma-separated)")
            
            if additional_skills:
                skills = skills + [s.strip() for s in additional_skills.split(",") if s.strip()]
            
            # Other fields
            col1, col2 = st.columns(2)
            
            with col1:
                experience = st.selectbox(
                    "Experience Level",
                    options=["Entry-level", "Junior", "Mid-level", "Senior", "Lead", "Manager", "Director", "Executive"],
                    index=2 if not details.get("experience_level") else ["Entry-level", "Junior", "Mid-level", "Senior", "Lead", "Manager", "Director", "Executive"].index(details.get("experience_level"))
                )
                
                location = st.selectbox(
                    "Work Location",
                    options=["Remote", "Hybrid", "On-site", "Flexible"],
                    index=0 if not details.get("location") else ["Remote", "Hybrid", "On-site", "Flexible"].index(details.get("location"))
                )
                
            with col2:
                budget_options = [
                    "Competitive", 
                    "$40,000 - $60,000",
                    "$60,000 - $80,000",
                    "$80,000 - $100,000",
                    "$100,000 - $120,000",
                    "$120,000 - $150,000",
                    "$150,000 - $200,000",
                    "$200,000+"
                ]
                
                budget = st.selectbox(
                    "Salary Range",
                    options=budget_options,
                    index=0 if not details.get("budget") else budget_options.index(details.get("budget")) if details.get("budget") in budget_options else 0
                )
                
                timeline_options = [
                    "As soon as possible",
                    "Within 2 weeks",
                    "Within 1 month",
                    "Within 2 months",
                    "Within 3 months",
                    "Next quarter"
                ]
                
                timeline = st.selectbox(
                    "Hiring Timeline",
                    options=timeline_options,
                    index=0 if not details.get("timeline") else timeline_options.index(details.get("timeline")) if details.get("timeline") in timeline_options else 0
                )
            
            # Notes field
            notes = st.text_area("Additional Notes", value=details.get("notes", ""))
            
            # Submit button
            submitted = st.form_submit_button("Save Details")
            if submitted:
                # Update position details
                updated_details = {
                    "skills": skills,
                    "experience_level": experience,
                    "location": location,
                    "budget": budget,
                    "timeline": timeline,
                    "notes": notes
                }
                
                # Add to conversation
                add_to_conversation("user", f"The {position} position requires these skills: {', '.join(skills)}. " +
                                          f"Experience level: {experience}. Location: {location}. " +
                                          f"Budget: {budget}. Timeline: {timeline}.")
                
                add_to_conversation("assistant", f"Thanks, I've updated the details for the {position} position. " +
                                               f"Would you like me to create a job description now?")
                
                # Save position details
                st.session_state.position_details[position] = updated_details
                
                # Add to FAISS if enabled
                if RAG_ENABLED and embedding_model is not None:
                    add_to_faiss(
                        f"Position: {position}\nSkills: {', '.join(skills)}\nExperience: {experience}\n" +
                        f"Location: {location}\nBudget: {budget}\nTimeline: {timeline}\nNotes: {notes}",
                        {
                            "type": "position_details",
                            "position": position
                        }
                    )
                
                # Save to session state
                if st.session_state.active_session_id:
                    save_session_state(st.session_state.active_session_id, {
                        "current_position": position,
                        "position_details": st.session_state.position_details,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                st.success("Position details saved successfully!")
        
        # Display current details in a formatted card
        st.subheader("Current Position Details")
        
        if details:
            st.markdown(f"""
            ### {position}
            
            **Skills**: {', '.join(details.get('skills', ['None specified']))}
            
            **Experience Level**: {details.get('experience_level', 'Not specified')}
            
            **Location**: {details.get('location', 'Not specified')}
            
            **Salary Range**: {details.get('budget', 'Not specified')}
            
            **Hiring Timeline**: {details.get('timeline', 'Not specified')}
            
            **Notes**: {details.get('notes', 'None')}
            """)
        else:
            st.info("No details saved yet. Please fill out the form above.")
    
    # Job Description tab
    elif st.session_state.active_tab == "Job Description":
        st.header("Job Description Generator")
        
        if not st.session_state.active_session_id:
            st.warning("Please start a new session from the Dashboard.")
            return
        
        position = st.session_state.current_position
        details = st.session_state.position_details.get(position, {})
        
        # Check if we already have job description
        existing_job_desc = st.session_state.job_descriptions.get(position, "")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Job Description", type="primary") or existing_job_desc:
                # Generate job description if not exists
                if not existing_job_desc:
                    with st.spinner("Generating job description..."):
                        job_desc = generate_job_description(position, details)
                    st.session_state.job_descriptions[position] = job_desc
                    
                    # Add to conversation
                    add_to_conversation("user", f"Can you create a job description for the {position} position?")
                    add_to_conversation("assistant", f"I've created a job description for the {position} position. You can view and edit it in the Job Description tab.")
                    
                    # Add to FAISS if enabled
                    if RAG_ENABLED and embedding_model is not None:
                        add_to_faiss(
                            job_desc,
                            {
                                "type": "job_description",
                                "position": position
                            }
                        )
                    
                    # Save job description to file
                    job_desc_dir = os.path.join(DATA_DIR, "job_descriptions")
                    os.makedirs(job_desc_dir, exist_ok=True)
                    job_desc_path = os.path.join(job_desc_dir, f"{position.replace(' ', '_').lower()}_job_description.md")
                    try:
                        with open(job_desc_path, 'w') as f:
                            f.write(job_desc)
                    except Exception as e:
                        st.error(f"Error saving job description to file: {str(e)}")
                    
                    # Save to session state
                    if st.session_state.active_session_id:
                        save_session_state(st.session_state.active_session_id, {
                            "current_position": position,
                            "job_descriptions": st.session_state.job_descriptions,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    st.success("Job description generated successfully!")
                else:
                    job_desc = existing_job_desc
                
                # Display job description with editor
                edited_job_desc = st.text_area(
                    "Edit Job Description",
                    value=job_desc,
                    height=500
                )
                
                # Save edits
                if edited_job_desc != job_desc:
                    st.session_state.job_descriptions[position] = edited_job_desc
                    
                    # Save job description to file
                    job_desc_dir = os.path.join(DATA_DIR, "job_descriptions")
                    os.makedirs(job_desc_dir, exist_ok=True)
                    job_desc_path = os.path.join(job_desc_dir, f"{position.replace(' ', '_').lower()}_job_description.md")
                    try:
                        with open(job_desc_path, 'w') as f:
                            f.write(edited_job_desc)
                    except Exception as e:
                        st.error(f"Error saving job description to file: {str(e)}")
                    
                    # Save edits to session state
                    if st.session_state.active_session_id:
                        save_session_state(st.session_state.active_session_id, {
                            "current_position": position,
                            "job_descriptions": st.session_state.job_descriptions,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    # Add to FAISS if enabled
                    if RAG_ENABLED and embedding_model is not None:
                        add_to_faiss(
                            edited_job_desc,
                            {
                                "type": "job_description",
                                "position": position,
                                "edited": True
                            }
                        )
                    
                    st.success("Job description updated!")
            else:
                st.info("Click the button to generate a job description based on the position details.")
        
        with col2:
            if details:
                st.markdown("### Position Summary")
                st.markdown(f"**Position**: {position}")
                st.markdown(f"**Skills**: {', '.join(details.get('skills', ['None specified'])[:3])}...")
                st.markdown(f"**Experience**: {details.get('experience_level', 'Not specified')}")
                st.markdown(f"**Location**: {details.get('location', 'Not specified')}")
                st.markdown(f"**Budget**: {details.get('budget', 'Not specified')}")
                
                # Similar job descriptions from FAISS
                if RAG_ENABLED and embedding_model is not None:
                    st.markdown("### Similar Job Descriptions")
                    results = search_faiss(f"job description {position}", doc_type="job_description", k=2)
                    if results:
                        for i, result in enumerate(results, 1):
                            doc = result["document"]
                            similarity = result["similarity"]
                            st.markdown(f"**Similar Position {i}** ({similarity:.2f} relevance)")
                            st.markdown(f"*{doc['metadata'].get('position', 'Unknown position')}*")
                            preview = doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
                            st.markdown(preview)
                    else:
                        st.info("No similar job descriptions found.")
            else:
                st.warning("Please complete the position details first.")
    
    # Hiring Plan tab
    elif st.session_state.active_tab == "Hiring Plan":
        st.header("Hiring Plan Generator")
        
        if not st.session_state.active_session_id:
            st.warning("Please start a new session from the Dashboard.")
            return
        
        position = st.session_state.current_position
        details = st.session_state.position_details.get(position, {})
        
        # Check if we already have hiring plan
        existing_plan = st.session_state.hiring_plans.get(position, "")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Hiring Plan", type="primary") or existing_plan:
                # Generate hiring plan if not exists
                if not existing_plan:
                    with st.spinner("Generating hiring plan..."):
                        hiring_plan = generate_hiring_plan(position, details)
                    st.session_state.hiring_plans[position] = hiring_plan
                    
                    # Add to conversation
                    add_to_conversation("user", f"Can you create a hiring plan for the {position} position?")
                    add_to_conversation("assistant", f"I've created a hiring plan for the {position} position. You can view and edit it in the Hiring Plan tab.")
                    
                    # Add to FAISS if enabled
                    if RAG_ENABLED and embedding_model is not None:
                        add_to_faiss(
                            hiring_plan,
                            {
                                "type": "hiring_plan",
                                "position": position
                            }
                        )
                    
                    # Save hiring plan to file
                    plan_dir = os.path.join(DATA_DIR, "hiring_plans")
                    os.makedirs(plan_dir, exist_ok=True)
                    plan_path = os.path.join(plan_dir, f"{position.replace(' ', '_').lower()}_hiring_plan.md")
                    try:
                        with open(plan_path, 'w') as f:
                            f.write(hiring_plan)
                    except Exception as e:
                        st.error(f"Error saving hiring plan to file: {str(e)}")
                    
                    # Save to session state
                    if st.session_state.active_session_id:
                        save_session_state(st.session_state.active_session_id, {
                            "current_position": position,
                            "hiring_plans": st.session_state.hiring_plans,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    st.success("Hiring plan generated successfully!")
                else:
                    hiring_plan = existing_plan
                
                # Display hiring plan with editor
                edited_plan = st.text_area(
                    "Edit Hiring Plan",
                    value=hiring_plan,
                    height=500
                )
                
                # Save edits
                if edited_plan != hiring_plan:
                    st.session_state.hiring_plans[position] = edited_plan
                    
                    # Save hiring plan to file
                    plan_dir = os.path.join(DATA_DIR, "hiring_plans")
                    os.makedirs(plan_dir, exist_ok=True)
                    plan_path = os.path.join(plan_dir, f"{position.replace(' ', '_').lower()}_hiring_plan.md")
                    try:
                        with open(plan_path, 'w') as f:
                            f.write(edited_plan)
                    except Exception as e:
                        st.error(f"Error saving hiring plan to file: {str(e)}")
                    
                    # Save edits to session state
                    if st.session_state.active_session_id:
                        save_session_state(st.session_state.active_session_id, {
                            "current_position": position,
                            "hiring_plans": st.session_state.hiring_plans,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    # Add to FAISS if enabled
                    if RAG_ENABLED and embedding_model is not None:
                        add_to_faiss(
                            edited_plan,
                            {
                                "type": "hiring_plan",
                                "position": position,
                                "edited": True
                            }
                        )
                    
                    st.success("Hiring plan updated!")
            else:
                st.info("Click the button to generate a hiring plan based on the position details.")
        
        with col2:
            if details:
                st.markdown("### Position Summary")
                st.markdown(f"**Position**: {position}")
                st.markdown(f"**Timeline**: {details.get('timeline', 'Not specified')}")
                
                # Next steps navigation
                st.markdown("### Next Steps")
                if st.button("Create Email Templates"):
                    st.session_state.active_tab = "Email Templates"
                    st.rerun()
                if st.button("Build Checklists"):
                    st.session_state.active_tab = "Checklists"
                    st.rerun()
                
                # Hiring plan status tracking
                st.markdown("### Process Status")
                st.info("Use the Checklists tab to track the progress of your hiring process.")
            else:
                st.warning("Please complete the position details first.")
    
    # Email Templates tab
    elif st.session_state.active_tab == "Email Templates":
        st.header("Email Templates")
        
        if not st.session_state.active_session_id:
            st.warning("Please start a new session from the Dashboard.")
            return
        
        position = st.session_state.current_position
        
        # Check if we already have email templates
        existing_emails = st.session_state.draft_emails.get(position, {})
        
        if st.button("Generate Email Templates", type="primary") or existing_emails:
            if not existing_emails:
                # Generate email templates
                email_types = [
                    {"type": "invitation", "recipient": "candidate"},
                    {"type": "rejection", "recipient": "candidate"},
                    {"type": "offer", "recipient": "candidate"},
                    {"type": "coordination", "recipient": "hiring manager"}
                ]
                
                with st.spinner("Generating email templates..."):
                    templates = {}
                    for email_config in email_types:
                        email_content = generate_email_template(
                            position, 
                            email_config["type"],
                            email_config["recipient"]
                        )
                        templates[f"{email_config['type']}_{email_config['recipient']}"] = email_content
                
                st.session_state.draft_emails[position] = templates
                
                # Add to conversation
                add_to_conversation("user", f"Can you create email templates for the {position} hiring process?")
                add_to_conversation("assistant", f"I've created email templates for the {position} hiring process. You can view and edit them in the Email Templates tab.")
                
                # Save to session state
                if st.session_state.active_session_id:
                    save_session_state(st.session_state.active_session_id, {
                        "current_position": position,
                        "draft_emails": st.session_state.draft_emails,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                # Save email templates to files
                email_dir = os.path.join(DATA_DIR, "email_templates", position.replace(' ', '_').lower())
                os.makedirs(email_dir, exist_ok=True)
                
                for email_name, content in templates.items():
                    template_path = os.path.join(email_dir, f"{email_name}.txt")
                    try:
                        with open(template_path, 'w') as f:
                            f.write(content)
                    except Exception as e:
                        st.error(f"Error saving email template to file: {str(e)}")
                
                # Add to FAISS if enabled
                if RAG_ENABLED and embedding_model is not None:
                    for email_name, content in templates.items():
                        email_type, recipient = email_name.split("_")
                        add_to_faiss(
                            content,
                            {
                                "type": "email_template",
                                "position": position,
                                "email_type": email_type,
                                "recipient": recipient
                            }
                        )
                
                st.success("Email templates generated successfully!")
                existing_emails = templates
            
            # Display email templates with tabs
            template_types = [
                ("invitation_candidate", "Interview Invitation"),
                ("rejection_candidate", "Rejection Email"),
                ("offer_candidate", "Offer Letter"),
                ("coordination_hiring manager", "Coordination")
            ]
            
            tabs = st.tabs([t[1] for t in template_types])
            
            for i, (template_key, template_name) in enumerate(template_types):
                with tabs[i]:
                    if template_key in existing_emails:
                        template_content = existing_emails[template_key]
                        
                        st.markdown(f"### {template_name} for {position}")
                        
                        edited_template = st.text_area(
                            f"Edit {template_name}",
                            value=template_content,
                            height=300,
                            key=f"email_{template_key}"
                        )
                        
                        # Save edits
                        if edited_template != template_content:
                            st.session_state.draft_emails[position][template_key] = edited_template
                            
                            # Save email template to file
                            email_dir = os.path.join(DATA_DIR, "email_templates", position.replace(' ', '_').lower())
                            os.makedirs(email_dir, exist_ok=True)
                            template_path = os.path.join(email_dir, f"{template_key}.txt")
                            try:
                                with open(template_path, 'w') as f:
                                    f.write(edited_template)
                            except Exception as e:
                                st.error(f"Error saving email template to file: {str(e)}")
                            
                            # Save edits to session state
                            if st.session_state.active_session_id:
                                save_session_state(st.session_state.active_session_id, {
                                    "current_position": position,
                                    "draft_emails": st.session_state.draft_emails,
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                            
                            # Add to FAISS if enabled
                            if RAG_ENABLED and embedding_model is not None:
                                email_type, recipient = template_key.split("_")
                                add_to_faiss(
                                    edited_template,
                                    {
                                        "type": "email_template",
                                        "position": position,
                                        "email_type": email_type,
                                        "recipient": recipient,
                                        "edited": True
                                    }
                                )
                            
                            st.success(f"{template_name} updated!")
                    else:
                        st.info(f"No {template_name} template generated yet.")
                    
                    # Add "Copy to Clipboard" button
                    if template_key in existing_emails:
                        if st.button(f"Copy {template_name} to Clipboard", key=f"copy_{template_key}"):
                            st.code(existing_emails[template_key])
                            st.success(f"{template_name} ready to copy!")
        else:
            st.info("Click the button to generate email templates for different stages of the hiring process.")
    
    # Checklists tab
    elif st.session_state.active_tab == "Checklists":
        st.header("Hiring Checklists")
        
        if not st.session_state.active_session_id:
            st.warning("Please start a new session from the Dashboard.")
            return
        
        position = st.session_state.current_position
        
        # Check if we already have checklists
        existing_checklists = st.session_state.checklists.get(position, {})
        
        if st.button("Generate Checklists", type="primary") or existing_checklists:
            if not existing_checklists:
                # Generate checklists
                checklist_types = ["screening", "interview", "onboarding"]
                
                with st.spinner("Generating checklists..."):
                    checklists = {}
                    for checklist_type in checklist_types:
                        checklist_items = generate_checklist(position, checklist_type)
                        checklists[checklist_type] = checklist_items
                
                st.session_state.checklists[position] = checklists
                
                # Add to conversation
                add_to_conversation("user", f"Can you create checklists for the {position} hiring process?")
                add_to_conversation("assistant", f"I've created checklists for the {position} hiring process. You can view and manage them in the Checklists tab.")
                
                # Save checklists to files
                checklist_dir = os.path.join(DATA_DIR, "checklists", position.replace(' ', '_').lower())
                os.makedirs(checklist_dir, exist_ok=True)
                
                for checklist_type, items in checklists.items():
                    checklist_path = os.path.join(checklist_dir, f"{checklist_type}_checklist.json")
                    try:
                        with open(checklist_path, 'w') as f:
                            json.dump(items, f, indent=2)
                    except Exception as e:
                        st.error(f"Error saving checklist to file: {str(e)}")
                
                # Save to session state
                if st.session_state.active_session_id:
                    save_session_state(st.session_state.active_session_id, {
                        "current_position": position,
                        "checklists": st.session_state.checklists,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                
                # Add to FAISS if enabled
                if RAG_ENABLED and embedding_model is not None:
                    for checklist_type, items in checklists.items():
                        add_to_faiss(
                            f"Checklist type: {checklist_type}, Items: {json.dumps(items)}",
                            {
                                "type": "checklist",
                                "position": position,
                                "checklist_type": checklist_type
                            }
                        )
                
                st.success("Checklists generated successfully!")
                existing_checklists = checklists
            
            # Display checklists with tabs
            checklist_names = {
                "screening": "Resume Screening",
                "interview": "Interview Process",
                "onboarding": "New Hire Onboarding"
            }
            
            tabs = st.tabs(list(checklist_names.values()))
            
            for i, (checklist_key, checklist_name) in enumerate(checklist_names.items()):
                with tabs[i]:
                    if checklist_key in existing_checklists:
                        checklist_items = existing_checklists[checklist_key]
                        
                        st.markdown(f"### {checklist_name} Checklist for {position}")
                        
                        # Convert to DataFrame for easier display and editing
                        df = pd.DataFrame(checklist_items)
                        
                        # Add a checkbox column if not exists
                        if "completed" not in df.columns:
                            df["completed"] = False
                        
                        # Display as an editable dataframe
                        edited_df = st.data_editor(
                            df,
                            column_config={
                                "item": "Task",
                                "category": "Category",
                                "timeline": "Timeline",
                                "assignee": "Assigned To",
                                "completed": st.column_config.CheckboxColumn("Completed")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Update checklist if changed
                        if not df.equals(edited_df):
                            st.session_state.checklists[position][checklist_key] = edited_df.to_dict('records')
                            
                            # Save checklist to file
                            checklist_dir = os.path.join(DATA_DIR, "checklists", position.replace(' ', '_').lower())
                            os.makedirs(checklist_dir, exist_ok=True)
                            checklist_path = os.path.join(checklist_dir, f"{checklist_key}_checklist.json")
                            try:
                                with open(checklist_path, 'w') as f:
                                    json.dump(edited_df.to_dict('records'), f, indent=2)
                            except Exception as e:
                                st.error(f"Error saving checklist to file: {str(e)}")
                            
                            # Save edits to session state
                            if st.session_state.active_session_id:
                                save_session_state(st.session_state.active_session_id, {
                                    "current_position": position,
                                    "checklists": st.session_state.checklists,
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                            
                            # Add to FAISS if enabled
                            if RAG_ENABLED and embedding_model is not None:
                                add_to_faiss(
                                    f"Updated checklist type: {checklist_key}, Items: {json.dumps(edited_df.to_dict('records'))}",
                                    {
                                        "type": "checklist",
                                        "position": position,
                                        "checklist_type": checklist_key,
                                        "edited": True
                                    }
                                )
                            
                            st.success(f"{checklist_name} checklist updated!")
                        
                        # Progress tracking
                        completed = sum(edited_df["completed"].astype(bool))
                        total = len(edited_df)
                        st.progress(completed / total if total > 0 else 0)
                        st.markdown(f"**Progress**: {completed}/{total} tasks completed ({completed/total*100:.0f}% complete)")
                        
                        # Export options
                        if st.button(f"Export {checklist_name} Checklist", key=f"export_{checklist_key}"):
                            # Create an Excel file for download
                            excel_file = f"{position}_{checklist_key}_checklist.xlsx"
                            
                            # Try to use Pandas Excel export
                            try:
                                edited_df.to_excel(excel_file, index=False)
                                
                                # Create download button
                                with open(excel_file, "rb") as file:
                                    st.download_button(
                                        label=f"Download {checklist_name} Checklist",
                                        data=file,
                                        file_name=excel_file,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            except Exception as e:
                                st.error(f"Error creating Excel file: {str(e)}")
                                
                                # Fallback to CSV
                                csv_file = f"{position}_{checklist_key}_checklist.csv"
                                edited_df.to_csv(csv_file, index=False)
                                
                                with open(csv_file, "rb") as file:
                                    st.download_button(
                                        label=f"Download {checklist_name} CSV",
                                        data=file,
                                        file_name=csv_file,
                                        mime="text/csv"
                                    )
                    else:
                        st.info(f"No {checklist_name} checklist generated yet.")
        else:
            st.info("Click the button to generate checklists for different stages of the hiring process.")
    
    # Knowledge Base tab
    elif st.session_state.active_tab == "Knowledge Base":
        st.header("Knowledge Base Search")
        
        # Search interface
        query = st.text_input("Search Query")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position_filter = st.text_input("Position Filter (optional)")
        
        with col2:
            doc_types = ["", "job_description", "hiring_plan", "email_template", "checklist", "position_details"]
            doc_type_filter = st.selectbox("Document Type Filter (optional)", doc_types)
        
        with col3:
            k = st.slider("Number of Results", 1, 10, 3)
        
        if st.button("Search") and query:
            if not RAG_ENABLED or embedding_model is None:
                st.error("Search functionality requires FAISS and sentence-transformers. Please install these libraries to enable search.")
            else:
                # Perform search
                with st.spinner("Searching knowledge base..."):
                    results = search_faiss(query, position=position_filter, doc_type=doc_type_filter, k=k)
                
                if results:
                    st.session_state.search_results = results
                    st.success(f"Found {len(results)} matches")
                else:
                    st.warning("No search results found.")
                    st.session_state.search_results = []
        
        # Display search results
        if "search_results" in st.session_state and st.session_state.search_results:
            st.subheader("Search Results")
            
            for i, result in enumerate(st.session_state.search_results, 1):
                doc = result["document"]
                metadata = doc.get("metadata", {})
                similarity = result["similarity"]
                
                with st.expander(f"Result {i}: {metadata.get('position', 'Unknown')} - {metadata.get('type', 'Unknown')} ({similarity:.2f} relevance)"):
                    st.markdown(f"**Position**: {metadata.get('position', 'Unknown')}")
                    st.markdown(f"**Type**: {metadata.get('type', 'Unknown')}")
                    st.markdown(f"**Created**: {metadata.get('timestamp', 'Unknown')[:10]}")
                    
                    # Display content based on type
                    content_type = metadata.get('type', 'unknown')
                    
                    if content_type == "job_description" or content_type == "hiring_plan":
                        st.markdown(doc["text"])
                    elif content_type == "email_template":
                        st.code(doc["text"])
                    elif content_type == "checklist":
                        try:
                            # Try to extract and display checklist items
                            import re
                            checklist_json = re.search(r'\[.*\]', doc["text"], re.DOTALL)
                            if checklist_json:
                                items = json.loads(checklist_json.group(0))
                                for j, item in enumerate(items, 1):
                                    st.markdown(f"{j}. **{item.get('item')}** - {item.get('category')}, {item.get('timeline')}")
                            else:
                                st.text_area("Content", value=doc["text"], height=200)
                        except:
                            st.text_area("Content", value=doc["text"], height=200)
                    else:
                        st.text_area("Content", value=doc["text"], height=200)
                    
                    # Quick action buttons based on type
                    if content_type == "job_description":
                        if st.button(f"Use This Job Description", key=f"use_jd_{i}"):
                            position_name = metadata.get('position', 'Unknown position')
                            if position_name in st.session_state.position_details:
                                st.session_state.job_descriptions[position_name] = doc["text"]
                                save_session_state(st.session_state.active_session_id, {
                                    "job_descriptions": st.session_state.job_descriptions,
                                    "timestamp": datetime.datetime.now().isoformat()
                                })
                                st.success(f"Job description applied to {position_name}")
                    elif content_type == "email_template":
                        if st.button(f"Copy Email Template", key=f"copy_email_{i}"):
                            st.code(doc["text"])
                            st.success("Email template ready to copy!")
        
        # Knowledge base statistics
        if RAG_ENABLED and embedding_model is not None:
            st.subheader("Knowledge Base Statistics")
            
            try:
                # Load FAISS index
                index, documents = init_faiss_index()
                
                if index is not None and documents:
                    # Count document types
                    doc_types = {}
                    positions = set()
                    
                    for doc in documents:
                        metadata = doc.get("metadata", {})
                        doc_type = metadata.get("type", "unknown")
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                        
                        position = metadata.get("position")
                        if position:
                            positions.add(position)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Documents", len(documents))
                        st.metric("Positions", len(positions))
                    
                    with col2:
                        # Create document type chart
                        df = pd.DataFrame({"Type": doc_types.keys(), "Count": doc_types.values()})
                        st.bar_chart(df.set_index("Type"))
            except Exception as e:
                st.error(f"Error loading knowledge base statistics: {str(e)}")
    
    # Export tab
    elif st.session_state.active_tab == "Export":
        st.header("Export Hiring Package")
        
        if not st.session_state.active_session_id:
            st.warning("Please start a new session from the Dashboard.")
            return
        
        position = st.session_state.current_position
        
        # Export options
        format_option = st.radio("Export Format", ["Markdown", "JSON"])
        format = "markdown" if format_option == "Markdown" else "json"
        
        if st.button("Export Hiring Package", type="primary"):
            # Check if we have enough data to export
            if position not in st.session_state.position_details:
                st.error("Position details are missing. Please complete the position details first.")
                return
            
            if position not in st.session_state.job_descriptions:
                st.warning("Job description is missing. It's recommended to create a job description before exporting.")
            
            if position not in st.session_state.hiring_plans:
                st.warning("Hiring plan is missing. It's recommended to create a hiring plan before exporting.")
            
            # Perform export
            with st.spinner("Exporting hiring package..."):
                export_path = export_position_data(position, format)
            
            if export_path and "Error" not in export_path:
                st.success(f"Hiring package exported successfully to {export_path}")
                
                # Create download button
                try:
                    with open(export_path, "rb") as file:
                        file_extension = ".md" if format == "markdown" else ".json"
                        file_name = f"{position.replace(' ', '_').lower()}_hiring_package{file_extension}"
                        
                        st.download_button(
                            label=f"Download {position} Hiring Package",
                            data=file,
                            file_name=file_name,
                            mime="text/plain" if format == "markdown" else "application/json"
                        )
                except Exception as e:
                    st.error(f"Error creating download: {str(e)}")
            else:
                st.error(f"Error exporting hiring package: {export_path}")
        
        # Preview section
        st.subheader("Hiring Package Preview")
        
        if position in st.session_state.position_details:
            details = st.session_state.position_details.get(position, {})
            job_desc = st.session_state.job_descriptions.get(position, "Not generated yet")
            hiring_plan = st.session_state.hiring_plans.get(position, "Not generated yet")
            
            preview_tabs = st.tabs(["Position Details", "Job Description", "Hiring Plan", "Email Templates", "Checklists"])
            
            with preview_tabs[0]:
                st.markdown(f"""
                ### {position} Details
                
                **Skills**: {', '.join(details.get('skills', ['None specified']))}
                
                **Experience Level**: {details.get('experience_level', 'Not specified')}
                
                **Location**: {details.get('location', 'Not specified')}
                
                **Salary Range**: {details.get('budget', 'Not specified')}
                
                **Hiring Timeline**: {details.get('timeline', 'Not specified')}
                """)
            
            with preview_tabs[1]:
                st.markdown(job_desc)
            
            with preview_tabs[2]:
                st.markdown(hiring_plan)
            
            with preview_tabs[3]:
                email_templates = st.session_state.draft_emails.get(position, {})
                if email_templates:
                    email_types = [
                        ("invitation_candidate", "Interview Invitation"),
                        ("rejection_candidate", "Rejection Email"),
                        ("offer_candidate", "Offer Letter"),
                        ("coordination_hiring manager", "Coordination")
                    ]
                    
                    for template_key, template_name in email_types:
                        if template_key in email_templates:
                            st.markdown(f"### {template_name}")
                            st.code(email_templates[template_key], language="text")
                else:
                    st.info("No email templates generated yet.")
            
            with preview_tabs[4]:
                checklists = st.session_state.checklists.get(position, {})
                if checklists:
                    checklist_names = {
                        "screening": "Resume Screening",
                        "interview": "Interview Process",
                        "onboarding": "New Hire Onboarding"
                    }
                    
                    for checklist_key, checklist_name in checklist_names.items():
                        if checklist_key in checklists:
                            st.markdown(f"### {checklist_name} Checklist")
                            
                            for i, item in enumerate(checklists[checklist_key], 1):
                                completed = "✅" if item.get("completed") else "⬜"
                                st.markdown(f"{completed} {i}. **{item.get('item')}** - {item.get('category')}, {item.get('timeline')}")
                else:
                    st.info("No checklists generated yet.")
        else:
            st.info("Complete the position details to see a preview.")

if __name__ == "__main__":
    main()