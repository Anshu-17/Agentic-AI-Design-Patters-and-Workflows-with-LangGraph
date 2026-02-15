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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.7
)
#------------------------ Define the state graph------------------------
class State(TypedDict):
    topic:str
    joke:str
    imporoved_joke:str
    final_joke:str

#------------------------ Define the functions------------------------

def generate_joke(state:State):
    """First call to llm to generate a joke"""
    msg = llm.invoke(f"write a short joke about: {state['topic']}")
    return {"joke":msg.content}
def improve_joke(state:State):
    """Second LLM call to improve the joke"""
    msg =  llm.invoke(f"Make this joke funnier by  adding wordplay or pun: {state['joke']}")
    return {"imporoved_joke":msg.content}

def polish_joke(state:State):
    """ Third LLM call to polish the joke"""
    msg =  llm.invoke(f"Add a surprising twist to the joke: {state['imporoved_joke']}")
    return {"final_joke":msg.content}

def check_punchline(state:State):
    """ Gate function to check if the joke has a punchline"""
    msg = llm.invoke(f"Is this joke funny? {state['final_joke']}")
    if "?" in state['joke'] or "!" in state['joke']:
        return "Pass"
    return "Fail"

#------------------------ Create a state graph------------------------

workflow = StateGraph(State)

#Adding Nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)
workflow.add_node("check_punchline", check_punchline)

#Adding edges
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline,{
    "Pass": "improve_joke",
    "Fail": END
    }
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)


# Compile
chain = workflow.compile()

# Save workflow as image
with open("graph.png", "wb") as f:
    f.write(chain.get_graph().draw_mermaid_png())
