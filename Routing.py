import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from  typing_extensions import  TypedDict,Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.7
)

class Route(BaseModel):
    step:Literal["joke","story", "poem"] = Field(None,description="The step to routing process")

router = llm.with_structured_output(Route)

#------------------------ Define the state graph------------------------

class State(TypedDict):
    input:str
    decision:str
    output:str

#------------------------Define the functions------------------------

#Nodes
def llm_call_1(state:State):
    """write a story"""
    print("write a story")
    result = llm.invoke(state["input"])
    return {"output":result.content}

def llm_call_2(state:State):
    """write a joke"""
    print("write a joke")
    result = llm.invoke(state["input"])
    return {"output":result.content}

def llm_call_3(state:State):
    """write a poem"""
    print("write a poem")
    result = llm.invoke(state["input"])
    return {"output":result.content}

def llm_call_router(state:State):
    """Run the augmented LLM with structured output to serve as routing logic"""
    decision = router.invoke(
        [
            SystemMessage(content = "Route the  input to story,joke, or poem based on user's request"),
            HumanMessage(content = state["input"])
        ]
    )
    return {"decision":decision.step}

# Conditional edge function to route to the appropriate node
def route_decision(state:State):
    """Route the workflow to the appropriate node based on the decision"""
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

#------------------------Create a state graph-------------------------
#build workflow
router_builder = StateGraph(State)

#Add nodes
router_builder.add_node("llm_call_1",llm_call_1)
router_builder.add_node("llm_call_2",llm_call_2)
router_builder.add_node("llm_call_3",llm_call_3)
router_builder.add_node("llm_call_router",llm_call_router)

#Add edges
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3"
    }
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

#compile
router_workflow = router_builder.compile()

#save workflow as image
with open("router_workflow.png", "wb") as f:
    f.write(router_workflow.get_graph().draw_mermaid_png())

#------------------------Run the workflow------------------------
state = router_workflow.invoke({"input": "write  me a story about quantum mechanics"})
print(state["output"])

