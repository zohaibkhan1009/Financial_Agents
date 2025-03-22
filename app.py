import streamlit as st
import os
from dotenv import load_dotenv
import openai  # Ensure OpenAI module is imported
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing API Key! Please set OPENAI_API_KEY in the .env file.")
    st.stop()

# Web Search Agent
web_search_agent = Agent(
    name="Web_Search_Agent",  # Adjusted name to meet pattern requirements
    role="Search the web for the information",
    model=OpenAIChat(model="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[DuckDuckGo()],
    instruction=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Finance AI Agent
finance_agent = Agent(
    name="Finance_AI_Agent",  # Adjusted name to meet pattern requirements
    model=OpenAIChat(model="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_news=True  # Removed invalid 'stock_fundamental'
        )
    ],
    instruction=["Use table to show the data"],
    show_tools_calls=True,
    markdown=True,
)

# Multi AI Agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use table to show the data"],
    show_tools_calls=True,
    markdown=True,
)

# Streamlit Web UI with Authentication
st.title("AI Multi-Agent System")

# User authentication
st.sidebar.header("User Authentication")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if login_button:
    if username == "admin" and password == "password123":
        st.session_state["authenticated"] = True
        st.success("Login successful!")
    else:
        st.session_state["authenticated"] = False
        st.error("Invalid credentials")

if st.session_state.get("authenticated", False):
    user_input = st.text_input("Ask a question:")
    if st.button("Get Response"):
        if user_input:
            try:
                response = multi_ai_agent.print_response(user_input, stream=True)
                st.write("### Response")
                st.write(response)
                
                # Display response in table format if applicable
                if isinstance(response, list) and all(isinstance(item, dict) for item in response):
                    st.table(response)
            except openai.OpenAIError as e:
                st.error(f"OpenAI API error: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.warning("Please log in to access the AI system.")
