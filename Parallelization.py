import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from  typing_extensions import  TypedDict
from langgraph.graph import StateGraph, START, END


# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.7
)

class State(TypedDict):
    topic:str
    joke:str
    story:str
    poem:str
    combined_output:str

#------------------------ Define the functions------------------------

# Nodes
def call_llm_1(state:State):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke":msg.content}

def call_llm_2(state:State):
    """Second LLM call to generate a story"""
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story":msg.content}

def call_llm_3(state:State):
    """Third LLM call to generate a poem"""
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return{"poem":msg.content}

def aggregator(state: State):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}

#------------------------Create a state graph------------------------

parallel_workflow_builder = StateGraph(State)

#Adding Nodes
parallel_workflow_builder.add_node("call_llm_1",call_llm_1)
parallel_workflow_builder.add_node("call_llm_2",call_llm_2)
parallel_workflow_builder.add_node("call_llm_3",call_llm_3)
parallel_workflow_builder.add_node("aggregator",aggregator)

#Adding Edges
parallel_workflow_builder.add_edge(START,"call_llm_1")
parallel_workflow_builder.add_edge(START,"call_llm_2")
parallel_workflow_builder.add_edge(START, "call_llm_3")
parallel_workflow_builder.add_edge("call_llm_1", "aggregator")
parallel_workflow_builder.add_edge("call_llm_2", "aggregator")
parallel_workflow_builder.add_edge("call_llm_3", "aggregator")
parallel_workflow_builder.add_edge("aggregator", END)

# Compile the workflow
parallel_workflow = parallel_workflow_builder.compile()

# Save workflow as image
with open("parallel_workflow.png", "wb") as f:
    f.write(parallel_workflow.get_graph().draw_mermaid_png())

#------------------------ Run the workflow------------------------
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])
