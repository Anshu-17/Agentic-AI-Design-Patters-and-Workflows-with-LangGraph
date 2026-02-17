import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from  typing_extensions import  TypedDict,Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated, List
import operator
from langgraph.types import Send

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.7
)

#Schema for structured output to use in planning
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="section of the report.",
    )

planner = llm.with_structured_output(Sections)

#------------------------Define the state graph-----------------------

#Graph State
class State(TypedDict):
    topic:str
    sections:list[Section]
    completed_report:Annotated[list,operator.add]
    final_report:str

#worker State
class WorkerState(TypedDict):
    section:Section
    completed_section:Annotated[list,operator.add]

#------------------------Define the functions------------------------
#Nodes

def orchestrator(state:State):
    """Orchestrator that generates a plan for the report"""
    report_sections = planner.invoke(
        [
            SystemMessage(content = "Generate a plan for the report"),
            HumanMessage(content = f"Here is the report topic: {state['topic']}")
        ]
    )
    return {"sections":report_sections.sections}

def llm_call(state:WorkerState):
    """Worker writes a section of the report"""
    section = llm.invoke(
        [
            SystemMessage(content = "write a report section"),
            HumanMessage(content = f"Here is the section name: {state['section'].name} and description: {state['section'].description}")
        ]
    )
    return {"completed_report": [section.content]}

def synthesizer(state:State):
    """Synthesize full report from sections"""
    completed_sections = state["completed_report"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report":completed_report_sections}

def assign_workers(state:State):
    """Assign a worker to each section in the plan"""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


#------------------------Buuild workflow -------------------------
#build workflow
orchestrator_worker_builder = StateGraph(State)

#Add nodes
orchestrator_worker_builder.add_node("orchestrator",orchestrator)
orchestrator_worker_builder.add_node("llm_call",llm_call)
orchestrator_worker_builder.add_node("synthesizer",synthesizer)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()

# Save workflow as image
with open("orchestrator_worker.png", "wb") as f:
    f.write(orchestrator_worker.get_graph().draw_mermaid_png())

#------------------------Run the workflow------------------------
state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
print(state["final_report"])
