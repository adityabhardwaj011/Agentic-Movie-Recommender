"""
agent.py, This is the AI "brain" we are building in agent.py. Its job is to talk to the user, understand what they want
and decide which specialist to assign the task to. The manager doesn't know how to find movies itself, 
but it's an expert at understanding requests and delegating(giving tasks)
"""

import os
from dotenv import load_dotenv # os and dotenv are used to securely load our secret OpenAI API key from the .env file.
from typing import TypedDict, Annotated # These are Python tools that help us define the structure of our agent's memory clearly.
import operator # to remember user inputs

from langchain_core.messages import BaseMessage # LangChain uses a standard format for messages (like HumanMessage, AIMessage). This import gives us access to that standard.
from langchain_core.tools import tool # decorator we can put on our Python functions to turn them into official "tools" that our agent can recognize and use.
from langgraph.graph import StateGraph, END # StateGraph is the blueprint for our agent's workflow, and END is a special marker that tells the agent when a turn is finished.
from langgraph.prebuilt import ToolNode # a pre-built "node" or step that is specifically designed to run tools. This saves us from writing a lot of boilerplate code. 
# from langchain_openai import ChatOpenAI # connect and communicate with an OpenAI model like GPT-4o.
from langchain_google_genai import ChatGoogleGenerativeAI # FREE :)

# Import our custom recommendation functions
from recommender_tool import get_recommendations_by_title, get_recommendations_by_description

# Load environment variables from.env, reads the OPENAI_API_KEY from it, and makes it available for our code to use.
load_dotenv() 

# DEFINING AGENT'S TOOLS

@tool # This decorator is what turns a normal Python function into a tool the AI can use
def movie_recommender_by_title(movie_title: str) -> list:
    """
    Recommends a list of movies similar to a SINGLE provided movie title.
    Use this tool when a user provides a specific movie title and asks for recommendations.
    """
    return get_recommendations_by_title(movie_title)

@tool
def movie_recommender_by_description(description: str) -> list: # The text inside the DOCSTRING(triple quotes) is not just a comment for humans. The Large Language Model (LLM) reads this description to understand what the tool does and when to use it.
    """
    Recommends movies based on a detailed text description. Use this for all general movie requests. 
    If a user specifies filters like a release year, genre, or country, include those details directly 
    in the 'description' text you provide to this tool to get the most relevant semantic matches. 
    For example, for 'a thriller after 2010', the description should be 'a thriller released after 2010'.
    """
    return get_recommendations_by_description(description)

# Toolbox we will give to our agent
tools = [movie_recommender_by_title, movie_recommender_by_description] # A py list of tools our agent can use

#2. DEFINING THE AGENT'S STATE(MEMORY)

class AgentState(TypedDict): # We are creating a blueprint for our memory, which will be structured like a Python dictionary.
    # just 1 key for our memory(i.e. messages) it will be a list and all the new messages should be added at end of the list no replace it
    messages: Annotated[list, operator.add] 

#3. DEFINING THE AGENT'S WORKFLOW (NODES AND EDGES)

# using LangGraph to define the agent's decision-making process.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # establishing our connection to gemini
llm_with_tools = llm.bind_tools(tools) # binding our tools to llm  giving it knowledge of our two movie recommender tools and allows it to decide when to call them.   

# First node(function) that performs one step
# This node's job is to take the current conversation history (state['messages']) and send it to the LLM and return response
def call_model(state: AgentState):  # The state: AgentState part is a modern Python feature called a type hint.
    response = llm_with_tools.invoke(state['messages']) 
    return {"messages": [response]}

# This is our second Node. It's a pre-built helper from LangGraph. Its job is to check if the LLM decided to use a tool. If it did, this node will automatically run the correct tool with the correct inputs
tool_executor = ToolNode(tools)

# This function defines a Conditional Edge (a decision-making arrow in our flowchart). It looks at the last message from the LLM.
def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    #  if the last_message contains any tool_calls, this line returns the string "tool" which tells the graph to move to the "tool" node next.
    if last_message.tool_calls:
        return "tool"
    return END # if there are no tool_calls means llm has the answer for user query and stops the current turn and sends the response

#4. CONSTRUCTING AND COMPILING THE GRAPH 

workflow = StateGraph(AgentState) # creates a new graph instance using StateGraph blueprint, initially an empty flowchart to be filled using AgentState as the structure for its memory.
workflow.add_node("agent", call_model) # making a node naming it "agent" and job is to run call_model function
workflow.add_node("tool", tool_executor) # 2nd node named "tool" and linked to tool_executor(pre-built) method
workflow.set_entry_point("agent") # This tells the graph where to begin its process, whenever new message comes the flow will start at the "agent" node, where the LLM will decide what to do.
workflow.add_conditional_edges("agent", should_continue) # adding conditional arrow from agent node,After the "agent"(or call_model)(LLM) has a response, the graph will run your should_continue function.
workflow.add_edge("tool", "agent") # unconditional arrow. After the tool node finishes running, the flow automatically goes back to the "agent" node. The agent (LLM) can then formulate a final, human-readable answer.

app = workflow.compile() # This command takes our entire defined graph and turns it into a runnable application. This is the final object we will use in our user interface to interact with our agent.