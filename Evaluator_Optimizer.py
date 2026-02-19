import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from  typing_extensions import Literal,TypedDict
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

#Graph State
class State(TypedDict):
    joke:str
    topic:str
    feedback:str
    funny_or_not:str

#Nodes
def llm_call_generator(state:State):
    """LLM Generates a joke"""
    if state.get("feedback"):
        msg = llm.invoke(f"write a joke {state['topic']} but take into account the feedback: {state['feedback']}")
    else:
        msg = llm.invoke(f"Write a joke about{state['topic']}")
        return {"joke":msg.content}

def llm_call_evaluator(state:State):
    """LLM Evaluates the joke"""
    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not":grade.grade,"feedback":grade.feedback}


# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


#------------------------Build workflow -------------------------
optimizer_builder = StateGraph(State)

#Add  nodes
optimizer_builder.add_node("llm_call_generator",llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator",llm_call_evaluator)

#Add Edges
optimizer_builder.add_edge(START,"llm_call_generator")
optimizer_builder.add_edge("llm_call_generator","llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        "Accepted":END,
        "Rejected + Feedback": "llm_call_generator"
    }
)

#compile the graph
optimizer_workflow = optimizer_builder.compile()

# Save workflow as image
with open("optimizer_workflow.png", "wb") as f:
    f.write(optimizer_workflow.get_graph().draw_mermaid_png())

#run the graph
state = optimizer_workflow.invoke({"topic":"cats"})
print(state["joke"])

