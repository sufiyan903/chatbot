import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Step 1: Retrieve API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Ensure OPENAI_API_KEY is set in the environment.")
if not GROQ_API_KEY:
    raise ValueError("Missing Groq API Key. Ensure GROQ_API_KEY is set in the environment.")

# Step 2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

search_tool = TavilySearchResults(max_results=2)

# Step 3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, groq_api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, openai_api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Invalid provider specified. Use 'Groq' or 'OpenAI'.")

    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages", [])
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1] if ai_messages else "No response from AI agent."