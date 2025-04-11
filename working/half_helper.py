import os
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Tuple, Optional, Union
import json
import re  # Add the 're' import at module level

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

# Step 2: Create the LLM - add your API key here or set as environment variable
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def extract_hiring_needs(state: HRState) -> HRState:
    """Extract the position that needs to be hired from the user's input using LLM."""
    print("Extracting hiring needs with LLM...")
    
    # Get the user's initial message
    user_input = ""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            user_input = message.content
            break
    
    print(f"User input: '{user_input}'")
    
    # Skip command line arguments or Jupyter kernel paths
    if user_input.startswith("-") or ".json" in user_input:
        position = "Software Developer"
        print(f"Input appears to be a command line argument. Using default position: {position}")
    else:
        # Use LLM to extract the position
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
        except Exception as e:
            print(f"Error extracting position with LLM: {str(e)}")
            position = "Software Developer"
            print(f"Using default position: {position}")
    
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
        "messages": state["messages"] + [
            AIMessage(content=f"I'll help you hire for the {position} position. Let's start by gathering some details.")
        ]
    }

def ask_clarifying_questions(state: HRState) -> HRState:
    """Ask a clarifying question using LLM based on missing details."""
    print("Asking clarifying question...")

    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {}).copy()

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

    # Prompt LLM to ask next question
    prompt = f"""
You're helping to collect information for a job posting for: {current_position}.

Missing info: {', '.join(missing_info) if missing_info else 'None'}

Conversation so far:
{conversation_text}

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
            return {
                **state,
                "clarification_complete": True,
                "stage": "job_desc",
                "messages": state["messages"] + [
                    AIMessage(content=f"Thanks! I think I have enough information to write a job description for the {current_position}.")
                ]
            }
        else:
            print(f"LLM generated question: {question}")
            return {
                **state,
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

    return {
        **state,
        "position_details": {
            **state["position_details"],
            current_position: current_details
        },
        "messages": state["messages"] + [
            AIMessage(content="Thanks, I've noted that. Let me see if I need to ask anything else.")
        ]
    }

def should_continue_clarifying(state: HRState) -> str:
    current_position = state["current_position"]
    details = state["position_details"].get(current_position, {})

    filled = sum(bool(v) for v in details.values())
    if filled >= 3:  # or however many you consider "enough"
        print("Enough information gathered.")
        return "move_to_job_desc"
    else:
        print("Still missing info. Continue clarifying.")
        return "continue_clarifying"

def human_input(state: HRState) -> HRState:
    """Function to collect human input in the workflow.
    
    This function pauses the workflow execution and prompts the user for input
    through the command line. The input is then added to the state as a HumanMessage.
    """
    current_position = state["current_position"]
    
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
    
    # Update the job description in the state
    job_descriptions = state["job_descriptions"].copy()
    job_descriptions[current_position] = job_description
    
    # Update the state
    return {
        **state,
        "job_descriptions": job_descriptions,
        "stage": "plan",
        "messages": state["messages"] + [
            AIMessage(content=f"Here's a job description for the {current_position} position:\n\n{job_description}\n\nNext, I'll create a hiring plan for this position.")
        ]
    }

def create_hiring_plan(state: HRState) -> HRState:
    """Create a hiring plan/checklist for the current position."""
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
    
    # Update the hiring plan in the state
    hiring_plans = state["hiring_plans"].copy()
    hiring_plans[current_position] = hiring_plan
    
    # Update the state
    return {
        **state,
        "hiring_plans": hiring_plans,
        "messages": state["messages"] + [
            AIMessage(content=f"Here's a hiring plan for the {current_position} position:\n\n{hiring_plan}")
        ]
    }


def perform_market_research(state: HRState) -> HRState:
    """Perform market research for the current position to gather industry insights."""
    print("Performing market research...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    
    # Build a search query based on position and details
    skills = ", ".join(position_details.get("skills", []))
    experience = position_details.get("experience_level", "")
    location = position_details.get("location", "")
    
    search_query = f"{current_position} salary ranges {experience} {location} {skills}"
    print(f"Search query: {search_query}")
    
    # Perform the search
    try:
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
        
        # Store the search results and summary
        updated_search_results = state.get("search_results", {})
        updated_search_results[current_position] = [
            {"type": "market_research", "query": search_query, "results": search_results, "summary": summary}
        ]
    except Exception as e:
        print(f"Error in market research: {str(e)}")
        summary = f"Unable to complete market research due to an error: {str(e)}"
        updated_search_results = state.get("search_results", {})
    
    # Update the state
    return {
        **state,
        "search_results": updated_search_results,
        "messages": state["messages"] + [
            AIMessage(content=f"I've researched market information for the {current_position} position.\n\n{summary}")
        ]
    }


def create_email_templates(state: HRState) -> HRState:
    """Create common email templates for the hiring process."""
    print("Creating email templates...")
    
    current_position = state["current_position"]
    position_details = state["position_details"].get(current_position, {})
    
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
            
            # Store the generated email
            generated_emails[f"{email_config['type']}_{email_config['recipient']}"] = email_content
            
            # Add this email to our list of messages to display
            email_content_messages.append(
                AIMessage(content=f"### {email_config['type'].title()} Email for {email_config['recipient'].title()}\n\n{email_content}")
            )
            
            # Print the email for debugging
            print(f"Generated {email_config['type']} email for {email_config['recipient']}:\n{email_content}\n")
            
        except Exception as e:
            print(f"Error generating {email_config['type']} email: {str(e)}")
            generated_emails[f"{email_config['type']}_{email_config['recipient']}"] = f"Error generating email: {str(e)}"
            email_content_messages.append(
                AIMessage(content=f"Error generating {email_config['type']} email for {email_config['recipient']}: {str(e)}")
            )
    
    # Update the draft emails in the state
    draft_emails = state.get("draft_emails", {})
    draft_emails[current_position] = generated_emails
    
    # Create an intro message
    intro_message = AIMessage(content=f"I've created the following email templates for the {current_position} position:")
    
    # Update the state with all our messages
    return {
        **state,
        "draft_emails": draft_emails,
        "messages": state["messages"] + [intro_message] + email_content_messages
    }

# 4.6 Integration function to build custom checklists for the current position
def build_position_checklists(state: HRState) -> HRState:
    """Build custom checklists for different stages of the hiring process."""
    print("Building custom checklists...")
    
    current_position = state["current_position"]
    
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
            # Generate the checklist - use invoke instead of direct call
            checklist = build_custom_checklist.invoke({
                "position": current_position,
                "checklist_type": checklist_config["type"],
                "specific_items": []  # No specific items required
            })
            
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
    
    # Update the checklists in the state
    checklists = state.get("checklists", {})
    checklists[current_position] = generated_checklists
    
    # Create an intro message
    intro_message = AIMessage(content=f"I've created the following detailed checklists for the {current_position} hiring process:")
    
    # Update the state with all our messages
    return {
        **state,
        "checklists": checklists,
        "messages": state["messages"] + [intro_message] + checklist_messages
    }

# 4.1 Market Research / Web Search Tool
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

# 4.2 Email Template Generator
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

# 4.3 Checklist Builder
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

# Add the new tool nodes
workflow.add_node("perform_market_research", perform_market_research)
workflow.add_node("create_email_templates", create_email_templates)
workflow.add_node("build_position_checklists", build_position_checklists)

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
        "search_results": {}
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
        
        # Add a summary message at the end
        completion_message = AIMessage(content=f"""
        I've completed all steps for hiring a {position}:
        """)
        
        return final_messages + [completion_message]
    except Exception as e:
        # Get detailed error information
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error running HR helper: {str(e)}")
        print(f"Traceback: {error_traceback}")
        
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

# Step 8: Create a simple chat interface
def chat_interface():
    """Simple chat interface for the HR helper."""
    
    print("HR Helper - Type 'exit' to quit")
    print("=" * 50)
    
    conversation = []
    user_input = input("You: ")
    
    while user_input.lower() != "exit":
        if not conversation:
            # First message, start the workflow
            messages = run_hr_helper(user_input)
            conversation = messages
        else:
            # Continue the conversation
            conversation.append(HumanMessage(content=user_input))
            
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
                "interaction_count": 0
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
    
    print("Thank you for using HR Helper!")

# Example usage
if __name__ == "__main__":
    import sys
    
    # Display version info
    print("Starting HR Helper application")
    
    if len(sys.argv) > 1:
        # Get input from command line
        user_input = " ".join(sys.argv[1:])
        messages = run_hr_helper(user_input)
        
        # Print only AI messages
        for message in messages:
            if isinstance(message, AIMessage):
                print(f"HR Helper: {message.content}")
    else:
        # Start interactive chat
        chat_interface()