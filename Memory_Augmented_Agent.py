import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated, List
import operator
from langgraph.types import Send

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# ---------------------- Memory Schema ----------------------
class MemoryItem(BaseModel):
    prompt: str
    response: str

class Memory(BaseModel):
    items: List[MemoryItem] = Field(
        description="Memory of previous interactions"
    )

# ---------------------- State Definitions ----------------------
class AgentState(TypedDict):
    topic: str
    memory: List[MemoryItem]
    completed_report: Annotated[list, operator.add]
    final_report: str

# Worker state for writing sections using memory
class WorkerState(TypedDict):
    section_name: str
    section_description: str
    memory: List[MemoryItem]
    completed_section: Annotated[list, operator.add]

# ---------------------- Functions ----------------------
def memory_orchestrator(state: AgentState):
    """Orchestrator generates a report plan and stores plan in memory"""
    planner = llm.with_structured_output(
        {
            "sections": List[dict]  # basic structured output
        }
    )

    plan = planner.invoke([
        SystemMessage(content="Generate a plan for a report."),
        HumanMessage(content=f"Report topic: {state['topic']}")
    ])

    # Update memory with the plan
    memory_update = [MemoryItem(prompt="Report Plan", response=str(plan))]
    state["memory"].extend(memory_update)
    return {"sections": plan["sections"], "memory": state["memory"]}

def memory_worker(state: WorkerState):
    """Worker writes a section using memory context"""
    # Construct memory context
    memory_context = "\n".join([f"Q: {m.prompt}\nA: {m.response}" for m in state["memory"]])
    
    response = llm.invoke([
        SystemMessage(content="Write a report section considering memory."),
        HumanMessage(content=f"Section name: {state['section_name']}\nDescription: {state['section_description']}\nMemory context:\n{memory_context}")
    ])
    
    # Update memory with new section
    new_memory = MemoryItem(prompt=f"Section: {state['section_name']}", response=response.content)
    state["memory"].append(new_memory)
    
    return {"completed_section": [response.content], "memory": state["memory"]}

def memory_synthesizer(state: AgentState):
    """Combine all completed sections into final report"""
    final_report = "\n\n---\n\n".join(state["completed_report"])
    return {"final_report": final_report}

def assign_memory_workers(state: AgentState):
    """Assign a worker to each planned section"""
    return [
        Send("memory_worker", {
            "section_name": s["name"],
            "section_description": s["description"],
            "memory": state["memory"]
        })
        for s in state["sections"]
    ]

# ---------------------- Build Workflow ----------------------
memory_agent_builder = StateGraph(AgentState)

# Add nodes
memory_agent_builder.add_node("memory_orchestrator", memory_orchestrator)
memory_agent_builder.add_node("memory_worker", memory_worker)
memory_agent_builder.add_node("memory_synthesizer", memory_synthesizer)

# Add edges
memory_agent_builder.add_edge(START, "memory_orchestrator")
memory_agent_builder.add_conditional_edges(
    "memory_orchestrator", assign_memory_workers, ["memory_worker"]
)
memory_agent_builder.add_edge("memory_worker", "memory_synthesizer")
memory_agent_builder.add_edge("memory_synthesizer", END)

memory_agent = memory_agent_builder.compile()

# Save workflow as image
with open("memory_agent.png", "wb") as f:
    f.write(memory_agent.get_graph().draw_mermaid_png())

# ---------------------- Run Workflow ----------------------
state = memory_agent.invoke({"topic": "Create a report on Memory-Augmented Agents", "memory": [], "completed_report": [], "final_report": ""})
print(state["final_report"])