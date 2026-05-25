from langgraph.graph import StateGraph, START, END
from src.state import MedicalSystemState
from src.agents import (
    diagnostician_independent_node,
    lab_analyst_independent_node,
    pharmacologist_independent_node,
    registrar_node,
    judge_node,
)

# ==========================================================
# ПОБУДОВА ГРАФА НЕЗАЛЕЖНОЇ АРХІТЕКТУРИ
# ==========================================================

workflow = StateGraph(MedicalSystemState)

workflow.add_node("registrar", registrar_node)
workflow.add_node("diagnostician", diagnostician_independent_node)
workflow.add_node("lab_analyst", lab_analyst_independent_node)
workflow.add_node("pharmacologist", pharmacologist_independent_node)
workflow.add_node("judge", judge_node)

workflow.add_edge(START, "registrar")

workflow.add_edge("registrar", "diagnostician")
workflow.add_edge("registrar", "lab_analyst")
workflow.add_edge("registrar", "pharmacologist")

workflow.add_edge("diagnostician", "judge")
workflow.add_edge("lab_analyst", "judge")
workflow.add_edge("pharmacologist", "judge")

workflow.add_edge("judge", END)
independent_app = workflow.compile()
