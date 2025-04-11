import streamlit as st
import os
import sys
from typing import List, Dict, Any
import json
import re

# Add the current directory to the path so we can import the paste module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import OpenAI for the LLM
from langchain_openai import ChatOpenAI

# Import necessary modules from the HR Helper
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load any environment variables
load_dotenv()

# Import the HR Helper workflow components
try:
    from half_helper import (
        HRState,
        extract_hiring_needs,
        ask_clarifying_questions,
        process_clarification_responses,
        should_continue_clarifying,
        create_job_description,
        create_hiring_plan,
        perform_market_research,
        create_email_templates,
        build_position_checklists,
    )
    print("Successfully imported from paste.py")
except ImportError as e:
    st.error(f"Error importing from paste.py: {str(e)}")
    st.info("Please make sure the file paste.py is in the same directory as this app.py file.")
    st.stop()

# Set page config to make it look nice
st.set_page_config(
    page_title="HR Helper Assistant",
    page_icon="👩‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve the UI - Add early to ensure it's applied
st.markdown("""
<style>
    /* Add border radius to cards and chat messages */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-top: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
    }
    
    /* Add spacing between elements */
    .main > div {
        padding-bottom: 1rem;
    }
    
    /* Fix tab layout issues */
    .stTabs {
        margin-top: 30px;
        border-top: 1px solid #ddd;
        padding-top: 15px;
    }
    
    /* Add space between tabs and content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    
    /* Improve the empty state styling */
    .empty-notice {
        padding: 30px;
        text-align: center;
        color: #888;
        font-style: italic;
        background: #f7f7f7;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* Make buttons more prominent */
    .stButton > button {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Add clear separation between sections */
    .section-divider {
        margin: 30px 0;
        border-bottom: 1px solid #ddd;
    }
    
    /* Fix for duplicate tabs */
    .stMarkdown + div [data-testid="stTabsTabListContainer"] {
        display: none !important;
    }
    
    /* Hide the duplicate tab bar at the bottom */
    div:has(+ footer) > .stTabs {
        display: none !important;
    }
    
    /* Make the main tabs stand out */
    .main-tabbed-content [data-testid="stTabsTabListContainer"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 5px 5px 0 5px;
    }
    
    /* Better styling for tab headers */
    .resource-header {
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Add proper spacing after the chat and before tabs */
    .content-divider {
        margin: 2rem 0;
        border-top: 1px solid #eee;
    }
    
    /* Force Streamlit to use dark theme colors for chat */
    .stChatMessage [data-testid="StyledTheme"] {
        background-color: transparent !important;
    }
    
    /* Hide default Streamlit chat avatars */
    .stChatMessage [data-testid="stImage"] {
        display: none !important;
    }
    
    /* User message styling - for dark theme */
    .user-message {
        background-color: #2C3333 !important;
        color: #ffffff !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin: 5px 0 5px auto !important;
        max-width: 80% !important;
        text-align: right !important;
        float: right !important;
        clear: both !important;
    }
    
    /* Assistant message styling - for dark theme */
    .assistant-message {
        background-color: #1E293B !important;
        color: #ffffff !important;
        border: 1px solid #4B5563 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin: 5px auto 5px 0 !important;
        max-width: 80% !important;
        text-align: left !important;
        float: left !important;
        clear: both !important;
    }
    
    /* Make links in messages more visible on dark background */
    .user-message a, .assistant-message a {
        color: #3B82F6 !important;
        text-decoration: underline !important;
    }
    
    /* Add subtle message indicators */
    .user-message::before {
        content: 'You' !important;
        display: block !important;
        font-size: 0.8em !important;
        font-weight: bold !important;
        margin-bottom: 5px !important;
        color: #3B82F6 !important;
    }
    
    .assistant-message::before {
        content: 'HR Assistant' !important;
        display: block !important;
        font-size: 0.8em !important;
        font-weight: bold !important;
        margin-bottom: 5px !important;
        color: #10B981 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize LLM if not already created
@st.cache_resource
def init_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

# Build and compile the HR Helper workflow
@st.cache_resource
def build_hr_workflow():
    # Create the workflow graph
    workflow = StateGraph(HRState)

    # Add nodes
    workflow.add_node("extract_hiring_needs", extract_hiring_needs)
    workflow.add_node("ask_clarifying_questions", ask_clarifying_questions)
    workflow.add_node("process_clarification_responses", process_clarification_responses)
    workflow.add_node("create_job_description", create_job_description)
    workflow.add_node("create_hiring_plan", create_hiring_plan)
    workflow.add_node("perform_market_research", perform_market_research)
    workflow.add_node("create_email_templates", create_email_templates)
    workflow.add_node("build_position_checklists", build_position_checklists)
    workflow.add_node("END", lambda state: state)

    # Add conditional edges
    workflow.add_conditional_edges(
        "process_clarification_responses",
        should_continue_clarifying,
        {
            "continue_clarifying": "ask_clarifying_questions",
            "move_to_job_desc": "create_job_description"
        }
    )

    # Define the workflow path
    workflow.set_entry_point("extract_hiring_needs")
    workflow.add_edge("extract_hiring_needs", "ask_clarifying_questions")
    workflow.add_edge("ask_clarifying_questions", "process_clarification_responses")
    workflow.add_edge("create_job_description", "create_hiring_plan")
    workflow.add_edge("create_hiring_plan", "perform_market_research")
    workflow.add_edge("perform_market_research", "create_email_templates")
    workflow.add_edge("create_email_templates", "build_position_checklists")
    workflow.add_edge("build_position_checklists", "END")

    # Compile and return the workflow
    return workflow.compile()

# Custom version of process_clarification_responses without adding a message
def streamlit_process_clarification(state: HRState) -> HRState:
    """Process user's latest answer to a clarifying question without adding an AI message."""
    print("Processing clarification response...")

    latest_response = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    current_position = state["current_position"]
    current_details = state["position_details"].get(current_position, {}).copy()

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
    ]) | init_llm() | parser

    try:
        extracted = chain.invoke({})

        for key, value in extracted.items():
            if value:
                current_details[key] = value
                print(f"Extracted {key}: {value}")

    except Exception as e:
        print(f"Failed to parse clarification: {e}")

    # Return without adding a message
    return {
        **state,
        "position_details": {
            **state["position_details"],
            current_position: current_details
        }
    }

# Custom functions that don't add content to messages
def streamlit_create_job_description(state: HRState) -> HRState:
    """Create a job description for the current position without adding to messages."""
    print("Creating job description...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    
    # Ensure we have some skills
    if not position_details.get("skills"):
        position_details = position_details.copy()
        position_details["skills"] = ["relevant technical skills", "problem-solving abilities", "communication skills"]
    
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
            
            Please create a complete job description using this information.""")
        ])
        
        job_desc_chain = job_desc_prompt | init_llm()
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
    
    # Update the job description in the state without adding to messages
    job_descriptions = state["job_descriptions"].copy()
    job_descriptions[current_position] = job_description
    
    # Return updated state without adding a message
    return {
        **state,
        "job_descriptions": job_descriptions,
        "stage": "plan"
    }

# Modified create_hiring_plan function
def streamlit_create_hiring_plan(state: HRState) -> HRState:
    """Create a hiring plan/checklist for the current position without adding to messages."""
    print("Creating hiring plan...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    
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
            
            Please create a comprehensive hiring plan and checklist.""")
        ])
        
        plan_chain = plan_prompt | init_llm()
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
    
    # Update the hiring plan in the state without adding to messages
    hiring_plans = state["hiring_plans"].copy()
    hiring_plans[current_position] = hiring_plan
    
    # Return updated state without adding a message
    return {
        **state,
        "hiring_plans": hiring_plans
    }

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "hr_state" not in st.session_state:
    st.session_state["hr_state"] = {
        "messages": [],
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
        "search_results": {}
    }

if "workflow_running" not in st.session_state:
    st.session_state["workflow_running"] = False

if "current_position" not in st.session_state:
    st.session_state["current_position"] = ""

if "needs_rerun" not in st.session_state:
    st.session_state["needs_rerun"] = False

# Initialize the LLM and workflow
llm = init_llm()
hr_helper = build_hr_workflow()

# Create the main layout
st.title("👩‍💼 HR Hiring Assistant")
st.write("I can help you create job descriptions, hiring plans, and more. Just tell me what position you're looking to hire for!")

# Create the sidebar with workflow status
with st.sidebar:
    st.header("Workflow Status")
    
    if st.session_state["current_position"]:
        st.subheader(f"Current Position: {st.session_state['current_position']}")
    
    # UPDATED PROGRESS SECTION
    st.subheader("Progress")
    workflow_stages = ["clarify", "job_desc", "plan", "complete"]
    stage_names = {
        "clarify": "Gathering Requirements",
        "job_desc": "Creating Job Description",
        "plan": "Building Hiring Plan",
        "complete": "Process Complete"
    }

    # Determine the current stage based on what has been generated
    current_stage = st.session_state["hr_state"].get("stage", "clarify")
    position = st.session_state["current_position"]

    # Update the stage based on what's been generated
    if position and "job_descriptions" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["job_descriptions"]:
        if st.session_state["hr_state"]["job_descriptions"][position]:
            # If job description exists, we've moved past that stage
            if current_stage == "job_desc":
                current_stage = "plan"
                
    if position and "hiring_plans" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["hiring_plans"]:
        if st.session_state["hr_state"]["hiring_plans"][position]:
            # If hiring plan exists, we've moved to complete stage
            current_stage = "complete"
            # Update the state to reflect completion
            st.session_state["hr_state"]["stage"] = "complete"

    current_index = workflow_stages.index(current_stage) if current_stage in workflow_stages else 0

    # Display the workflow progress
    for i, stage in enumerate(workflow_stages):
        if i < current_index:
            st.success(stage_names[stage])
        elif i == current_index:
            st.info(stage_names[stage] + " (Current)")
        else:
            st.text(stage_names[stage])
    
    if "position_details" in st.session_state["hr_state"] and st.session_state["current_position"]:
        position = st.session_state["current_position"]
        details = st.session_state["hr_state"]["position_details"].get(position, {})
        
        st.subheader("Position Details")
        
        # Display the details in a more structured way
        detail_data = []
        
        if details.get("skills"):
            skills_str = ", ".join(details.get("skills", []))
            detail_data.append(("Skills", skills_str))
        
        if details.get("budget"):
            detail_data.append(("Budget", str(details.get("budget", "Not specified"))))
        
        if details.get("experience_level"):
            detail_data.append(("Experience", str(details.get("experience_level", "Not specified"))))
        
        if details.get("location"):
            detail_data.append(("Location", str(details.get("location", "Not specified"))))
        
        if details.get("timeline"):
            detail_data.append(("Timeline", str(details.get("timeline", "Not specified"))))
        
        # Display in a cleaner format
        for label, value in detail_data:
            st.write(f"**{label}:** {value}")
    
    # Add a reset button
    if st.button("Start New Hiring Process"):
        st.session_state["messages"] = []
        st.session_state["hr_state"] = {
            "messages": [],
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
            "search_results": {}
        }
        st.session_state["workflow_running"] = False
        st.session_state["current_position"] = ""
        st.session_state["needs_rerun"] = True

# Create a debug section in the sidebar
with st.sidebar:
    with st.expander("Debug Info"):
        if st.button("Print Current State"):
            st.write("Current Position:", st.session_state["current_position"])
            st.write("Stage:", st.session_state["hr_state"].get("stage", "None"))
            st.write("Clarification Complete:", st.session_state["hr_state"].get("clarification_complete", False))
            if "position_details" in st.session_state["hr_state"] and st.session_state["current_position"]:
                st.write("Position Details:", 
                         st.session_state["hr_state"]["position_details"].get(st.session_state["current_position"], {}))

# Check if a rerun is needed
if st.session_state["needs_rerun"]:
    st.session_state["needs_rerun"] = False
    st.rerun()

# Display the chat messages with custom alignment
chat_container = st.container()
with chat_container:
    # Display messages with CSS-based alignment
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Clear the floats at the end
    st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

# Only show the tabs when we've completed the clarification phase
# AND we're ready to generate resources or have already generated them
show_tabs = (
    st.session_state["workflow_running"] and 
    st.session_state["hr_state"].get("clarification_complete", False) and
    (
        st.session_state["hr_state"].get("stage") == "job_desc" or
        any(st.session_state["hr_state"].get("job_descriptions", {}).values())
    )
)

# Check if we need to display the generate button after answering questions
# Display the button in its own section, outside the tabs
if (st.session_state["workflow_running"] and 
    st.session_state["hr_state"].get("stage") == "job_desc" and 
    st.session_state["hr_state"].get("clarification_complete", False) and
    not st.session_state["hr_state"]["job_descriptions"].get(st.session_state["current_position"], "")):
    
    # Add visual separator
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Create a separate container for the button
    button_container = st.container()
    
    with button_container:
        st.markdown("### Ready to generate resources")
        st.write("I have all the information needed to create the job description and hiring resources.")
        
        if st.button("Generate Job Description and Hiring Plan", key="generate_button", use_container_width=True):
            with st.spinner("Generating job description, hiring plan, and other resources..."):
                # Use the modified functions that don't add to messages
                updated_state = streamlit_create_job_description(st.session_state["hr_state"])
                updated_state = streamlit_create_hiring_plan(updated_state)
                
                # Perform market research
                try:
                    updated_state = perform_market_research(updated_state)
                    # Filter out any market research messages
                    filtered_messages = [m for m in updated_state["messages"] 
                                      if not (isinstance(m, AIMessage) and 
                                             "researched market information" in m.content)]
                    updated_state["messages"] = filtered_messages
                except Exception as e:
                    st.warning(f"Market research encountered an issue: {str(e)}")
                
                # Create email templates
                try:
                    # Just extract the email templates without adding messages
                    email_types = [
                        {"type": "invitation", "recipient": "candidate", "purpose": "interview invitation"},
                        {"type": "rejection", "recipient": "candidate", "purpose": "polite rejection"},
                        {"type": "offer", "recipient": "candidate", "purpose": "job offer"},
                        {"type": "coordination", "recipient": "hiring manager", "purpose": "interview coordination"}
                    ]
                    generated_emails = {}
                    position = updated_state["current_position"]
                    position_details = updated_state["position_details"].get(position, {})
                    
                    for email_config in email_types:
                        custom_details = f"""
                        Position Details:
                        - Skills: {', '.join(position_details.get('skills', []))}
                        - Experience: {position_details.get('experience_level', 'Not specified')}
                        - Location: {position_details.get('location', 'Not specified')}
                        """
                        
                        try:
                            email_prompt = ChatPromptTemplate.from_messages([
                                SystemMessage(content="""You are an HR professional who writes excellent, professional emails.
                                Create an email template for the specified purpose.
                                Include subject line and body text.
                                Keep the tone professional but warm.
                                Use appropriate salutations and closings."""),
                                HumanMessage(content=f"""
                                Create an email template with the following specifications:
                                - Position: {position}
                                - Email Type: {email_config['type']} (invitation, rejection, offer, or follow-up)
                                - Recipient: {email_config['recipient']} (candidate, hiring manager, or team)
                                - Custom Details: {custom_details}
                                
                                Format as:
                                Subject: [Subject line]
                                
                                [Email body with appropriate placeholders]
                                
                                [Closing]
                                """)
                            ])
                            
                            email_chain = email_prompt | llm
                            email_result = email_chain.invoke({})
                            email_content = email_result.content
                            
                            generated_emails[f"{email_config['type']}_{email_config['recipient']}"] = email_content
                        except Exception as e:
                            generated_emails[f"{email_config['type']}_{email_config['recipient']}"] = f"Error generating email: {str(e)}"
                    
                    # Update draft emails without adding messages
                    draft_emails = updated_state.get("draft_emails", {})
                    draft_emails[position] = generated_emails
                    updated_state["draft_emails"] = draft_emails
                except Exception as e:
                    st.warning(f"Email template generation encountered an issue: {str(e)}")
                
                # Build position checklists
                try:
                    # Generate checklists without adding messages
                    checklist_types = [
                        {"type": "screening", "description": "Resume screening process"},
                        {"type": "interview", "description": "Interview process management"},
                        {"type": "onboarding", "description": "New hire onboarding"}
                    ]
                    
                    generated_checklists = {}
                    position = updated_state["current_position"]
                    
                    for checklist_config in checklist_types:
                        try:
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
                                - Checklist Type: {checklist_config['type']}
                                - Required Items: Standard items for this type
                                
                                The checklist should include at least 5-10 items and cover all important aspects of {checklist_config['type']}.
                                """)
                            ])
                            
                            parser = JsonOutputParser()
                            checklist_chain = checklist_prompt | llm | parser
                            checklist_result = checklist_chain.invoke({})
                            
                            if isinstance(checklist_result, list):
                                generated_checklists[checklist_config['type']] = checklist_result
                            else:
                                generated_checklists[checklist_config['type']] = [
                                    {
                                        "item": f"Error parsing {checklist_config['type']} checklist",
                                        "category": "Error",
                                        "timeline": "N/A",
                                        "assignee": "HR",
                                        "completed": False
                                        }
                                ]
                        except Exception as e:
                            generated_checklists[checklist_config['type']] = [
                                {
                                    "item": f"Error creating {checklist_config['type']} checklist: {str(e)}",
                                    "category": "Error",
                                    "timeline": "N/A",
                                    "assignee": "HR",
                                    "completed": False
                                }
                            ]
                    
                    # Update checklists without adding messages
                    checklists = updated_state.get("checklists", {})
                    checklists[position] = generated_checklists
                    updated_state["checklists"] = checklists
                except Exception as e:
                    st.warning(f"Checklist generation encountered an issue: {str(e)}")
                
                # Update the state to indicate completion
                updated_state["stage"] = "complete"
                
                # Add a completion message to the chat
                completion_message = AIMessage(content=f"✅ Job description, hiring plan, and other resources have been generated! You can view them in the tabs below.")
                
                # Update state and session messages
                st.session_state["hr_state"] = updated_state
                st.session_state["hr_state"]["messages"].append(completion_message)
                st.session_state["messages"].append({"role": "assistant", "content": completion_message.content})
                
                # Set flag to rerun
                st.session_state["needs_rerun"] = True
                st.rerun()

# Only show tabs if we have completed the clarification phase and there's content to show
if show_tabs and st.session_state["current_position"]:
    # Create a visual separator and then a unique container for the tabs
    st.markdown('<div class="content-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-tabbed-content">', unsafe_allow_html=True)
    
    # Create this container only once
    tab_container = st.container()
    
    with tab_container:
        # Create a header with the position
        st.subheader(f"Resources for {st.session_state['current_position']} Position")
        
        # Create tabs - THIS IS THE ONLY PLACE TABS ARE CREATED IN THE CODE
        tabs = st.tabs(["Job Description", "Hiring Plan", "Email Templates", "Checklists", "Market Research"])
        
        position = st.session_state["current_position"]
        
        # Job Description Tab
        with tabs[0]:
            st.markdown('<p class="resource-header">Job Description</p>', unsafe_allow_html=True)
            if "job_descriptions" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["job_descriptions"]:
                job_desc = st.session_state["hr_state"]["job_descriptions"].get(position, "")
                if job_desc:
                    st.markdown(job_desc)
                    
                    # Download button for job description
                    st.download_button(
                        label="Download Job Description",
                        data=job_desc,
                        file_name=f"{position.replace(' ', '_')}_job_description.md",
                        mime="text/markdown"
                    )
                else:
                    st.info("Job description will appear here once created.")
            else:
                st.info("Job description will appear here once created.")
                
        # Hiring Plan Tab
        with tabs[1]:
            st.markdown('<p class="resource-header">Hiring Plan</p>', unsafe_allow_html=True)
            if "hiring_plans" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["hiring_plans"]:
                plan = st.session_state["hr_state"]["hiring_plans"].get(position, "")
                if plan:
                    st.markdown(plan)
                    
                    # Download button for hiring plan
                    st.download_button(
                        label="Download Hiring Plan",
                        data=plan,
                        file_name=f"{position.replace(' ', '_')}_hiring_plan.md",
                        mime="text/markdown"
                    )
                else:
                    st.info("Hiring plan will appear here once created.")
            else:
                st.info("Hiring plan will appear here once created.")
        
        # Email Templates Tab
        with tabs[2]:
            st.markdown('<p class="resource-header">Email Templates</p>', unsafe_allow_html=True)
            if "draft_emails" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["draft_emails"]:
                emails = st.session_state["hr_state"]["draft_emails"].get(position, {})
                
                if emails:
                    email_type = st.selectbox(
                        "Select Email Template:",
                        options=list(emails.keys()),
                        format_func=lambda x: x.replace("_", " ").title()
                    )
                    
                    if email_type:
                        st.markdown(emails.get(email_type, ""))
                        
                        # Download button for selected email
                        st.download_button(
                            label="Download Email Template",
                            data=emails.get(email_type, ""),
                            file_name=f"{position.replace(' ', '_')}_{email_type}_email.txt",
                            mime="text/plain"
                        )
                else:
                    st.info("Email templates will appear here once created.")
            else:
                st.info("Email templates will appear here once created.")
        
        # Checklists Tab
        with tabs[3]:
            st.markdown('<p class="resource-header">Checklists</p>', unsafe_allow_html=True)
            if "checklists" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["checklists"]:
                checklists = st.session_state["hr_state"]["checklists"].get(position, {})
                
                if checklists:
                    checklist_type = st.selectbox(
                        "Select Checklist:",
                        options=list(checklists.keys()),
                        format_func=lambda x: x.title(),
                        key="checklist_select"
                    )
                    
                    if checklist_type and checklist_type in checklists:
                        checklist_items = checklists[checklist_type]
                        
                        # Create a checkbox for each item
                        for i, item in enumerate(checklist_items, 1):
                            task = item.get("item", "Undefined task")
                            category = item.get("category", "General")
                            timeline = item.get("timeline", "Anytime")
                            assignee = item.get("assignee", "HR")
                            
                            st.checkbox(
                                f"{task} (Category: {category}, Timeline: {timeline}, Assignee: {assignee})",
                                key=f"checklist_{checklist_type}_{i}"
                            )
                else:
                    st.info("Checklists will appear here once created.")
            else:
                st.info("Checklists will appear here once created.")
        
        # Market Research Tab
        with tabs[4]:
            st.markdown('<p class="resource-header">Market Research</p>', unsafe_allow_html=True)
            if "search_results" in st.session_state["hr_state"] and position in st.session_state["hr_state"]["search_results"]:
                search_results = st.session_state["hr_state"]["search_results"].get(position, [])
                
                if search_results:
                    for result in search_results:
                        st.subheader(result.get("type", "Research").title())
                        st.write(f"**Query:** {result.get('query', 'No query')}")
                        
                        if "summary" in result:
                            st.markdown(result["summary"])
                        
                        if "results" in result:
                            with st.expander("View Raw Results"):
                                st.json(result["results"])
                else:
                    st.info("Market research results will appear here once completed.")
            else:
                st.info("Market research results will appear here once created.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process chat input
if prompt := st.chat_input("What position are you hiring for?"):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.spinner("Processing..."):
        if not st.session_state["workflow_running"]:
            # First message - initialize the workflow
            st.session_state["hr_state"]["messages"] = [HumanMessage(content=prompt)]
            
            try:
                # Extract hiring needs first
                initial_state = extract_hiring_needs(st.session_state["hr_state"])
                st.session_state["current_position"] = initial_state.get("current_position", "")
                
                # Ask first clarifying question
                result = ask_clarifying_questions(initial_state)
                
                # Update the state
                st.session_state["hr_state"] = result
                st.session_state["workflow_running"] = True
                
                # Show the AI messages (welcome and first question)
                for message in result["messages"]:
                    if isinstance(message, AIMessage):
                        response_content = message.content
                        st.session_state["messages"].append({"role": "assistant", "content": response_content})
            except Exception as e:
                error_msg = f"Error starting workflow: {str(e)}"
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
            
            # Rerun to update UI
            st.rerun()
        else:
            # Continue the conversation
            # Add the user's message to the workflow state
            st.session_state["hr_state"]["messages"].append(HumanMessage(content=prompt))
            
            try:
                # Process the user's response first without adding a message
                updated_state = streamlit_process_clarification(st.session_state["hr_state"])
                
                # Add acknowledgment message
                acknowledgment = f"Thanks for providing those details about the {st.session_state['current_position']} position."
                updated_state["messages"].append(AIMessage(content=acknowledgment))
                st.session_state["messages"].append({"role": "assistant", "content": acknowledgment})
                
                # Check if we need more clarification
                next_step = should_continue_clarifying(updated_state)
                
                # Only run one step at a time
                if next_step == "continue_clarifying":
                    # Ask another clarifying question
                    updated_state = ask_clarifying_questions(updated_state)
                    
                    # Find and display the new question
                    new_question = next((m.content for m in reversed(updated_state["messages"]) 
                                        if isinstance(m, AIMessage) and m.content != acknowledgment), "")
                    
                    if new_question:
                        st.session_state["messages"].append({"role": "assistant", "content": new_question})
                elif updated_state["stage"] == "clarify":
                    # Transition to job description stage
                    updated_state["stage"] = "job_desc"
                    updated_state["clarification_complete"] = True
                    
                    # Add a transition message
                    completion_msg = f"Great! I now have enough information to create the job description for the {updated_state['current_position']} position. Click the 'Generate Job Description and Hiring Plan' button when you're ready."
                    updated_state["messages"].append(AIMessage(content=completion_msg))
                    
                    st.session_state["messages"].append({"role": "assistant", "content": completion_msg})
                
                # Update the state
                st.session_state["hr_state"] = updated_state
                
                # Rerun to update UI
                st.rerun()
                
            except Exception as e:
                error_msg = f"Error processing response: {str(e)}"
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                st.rerun()

# # Add a footer with better styling
# st.markdown("""
# <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; text-align: center; color: #888; font-size: 0.8rem;">
#     <p>HR Hiring Assistant powered by LangGraph and Streamlit</p>
# </div>
# """, unsafe_allow_html=True)