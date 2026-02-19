import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from  typing_extensions import Literal,TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.7
)

#------------------------Defining the Tools------------------------

@tool
def multiply(a: float, b: float) -> float:
    """Multiply a and b.

    Args:
        a: first float
        b: second float
    """
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add a and b.

    Args:
        a: first float
        b: second float
    """
    return a + b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: first float
        b: second float
    """
    return a / b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a: first float
        b: second float
    """
    return a - b

@tool
def power(a: float, b: float) -> float:
    """Raise a to the power of b.

    Args:
        a: base
        b: exponent
    """
    return a ** b

tools = [multiply,add,divide,subtract,power]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

#------------------------definig Nodes------------------------

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state:dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation),tool_call_id=tool_call["id"]))
    return {"messages":result}

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call

def should_continue(state:MessagesState)->Literal["environment",END]:
    "Decide if we should continue the loop or stop based  upon whether the LLM made a tool call"
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "Action"
    else:
        return END

#------------------------Build Workflow-------------------------

agent_builder = StateGraph(MessagesState)

#Add nodes
agent_builder.add_node("llm_call",llm_call)
agent_builder.add_node("environment", tool_node)

#Add edges
agent_builder.add_edge(START,"llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action":"environment",
        END:END
    }
)
agent_builder.add_edge("environment","llm_call")

#compile
agent_workflow = agent_builder.compile()

#save workflow as image
with open("agent_workflow.png", "wb") as f:
    f.write(agent_workflow.get_graph().draw_mermaid_png())

#run the workflow
messages = [HumanMessage(content = "Add 2 and 5 and multiply the result by 4 and then divide the result by 10")]
message = agent_workflow.invoke({"messages":messages})
for m in message["messages"]:
    m.pretty_print()

    
    