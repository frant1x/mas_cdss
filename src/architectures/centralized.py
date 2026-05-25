from typing import List
from langgraph.graph import StateGraph, START, END

from src.state import MedicalSystemState
from src.agents import (
    registrar_node,
    diagnostician_centralized_node,
    lab_analyst_centralized_node,
    pharmacologist_centralized_node,
    orchestrator_node,
)


# ==========================================================
# 1. ДИНАМІЧНИЙ ЦЕНТРАЛІЗОВАНИЙ МАРШРУТИЗАТОР (FORK/JOIN)
# ==========================================================
def centralized_router(state: MedicalSystemState) -> List[str]:
    """
    Аналізує карту фідбеків Оркестратора та генерує динамічний набір цілей.
    """
    if state.discussion_turns >= 4:
        print("[Оркестратор] Досягнуто ліміту ітерацій безпеки. Примусове завершення.")
        return ["end_process"]

    if state.final_report is not None:
        return ["end_process"]

    next_destinations = []
    if state.diagnostician_feedback is not None:
        next_destinations.append("diagnostician")
    if state.lab_analyst_feedback is not None:
        next_destinations.append("lab_analyst")
    if state.pharmacologist_feedback is not None:
        next_destinations.append("pharmacologist")

    if not next_destinations:
        return ["end_process"]

    print(
        f"[Роутер Оркестратора] Направлено на паралельне виправлення: {next_destinations}"
    )
    return next_destinations


# ==========================================================
# 2. ПОБУДОВА ІТЕРАТИВНОГО ЦЕНТРАЛІЗОВАНОГО ГРАФА
# ==========================================================
workflow = StateGraph(MedicalSystemState)

workflow.add_node("registrar", registrar_node)
workflow.add_node("diagnostician", diagnostician_centralized_node)
workflow.add_node("lab_analyst", lab_analyst_centralized_node)
workflow.add_node("pharmacologist", pharmacologist_centralized_node)
workflow.add_node("orchestrator", orchestrator_node)

workflow.add_edge(START, "registrar")

workflow.add_edge("registrar", "diagnostician")
workflow.add_edge("registrar", "lab_analyst")
workflow.add_edge("registrar", "pharmacologist")

workflow.add_edge("diagnostician", "orchestrator")
workflow.add_edge("lab_analyst", "orchestrator")
workflow.add_edge("pharmacologist", "orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    centralized_router,
    {
        "diagnostician": "diagnostician",
        "lab_analyst": "lab_analyst",
        "pharmacologist": "pharmacologist",
        "end_process": END,
    },
)

centralized_app = workflow.compile()
