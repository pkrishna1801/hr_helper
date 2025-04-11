# HR Helper

A Streamlit-based application that helps HR professionals create job descriptions and hiring plans using AI.

## Features

- **Interactive Chat Interface**: Communicate with the HR Helper through a user-friendly chat interface
- **Job Description Generation**: Create detailed, professional job descriptions based on position details
- **Hiring Plan Creation**: Generate comprehensive hiring plans with checklists for each stage of the recruitment process
- **Workflow Visualization**: View the HR Helper workflow as a Mermaid diagram
- **Conversation History**: Save and review past conversations

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hr-helper.git
   cd hr-helper
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   TAVILY_API_KEY=your-tavily-api-key-here
   ```

4. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
5. Interactive session
   ```
   python hr_helper.py "I need to hire a machine learning intern. Can you help?"
   ```
## Usage

1. Open the application in your web browser (typically at http://localhost:8501)
2. Enter a description of the position you're hiring for in the text area
3. Respond to the HR Helper's clarifying questions about the position
4. Review the generated job description and hiring plan
5. Save the conversation if needed

## Project Structure

- `app.py`: Main Streamlit application
- `hr_helper_workflow.py`: Core HR Helper workflow logic using LangGraph
- `workflow_diagram.mmd`: Mermaid diagram of the HR Helper workflow
- `requirements.txt`: Python dependencies
- `.env`: Environment variables for API keys

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for developing applications powered by language models
- **LangGraph**: Library for building stateful, multi-actor applications with LLMs
- **OpenAI API**: For generating job descriptions and hiring plans
- **Mermaid**: For visualizing the workflow

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
