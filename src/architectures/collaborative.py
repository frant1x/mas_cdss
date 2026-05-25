from langgraph.graph import StateGraph, START, END

from src.state import MedicalSystemState
from src.agents import (
    registrar_node,
    diagnostician_collaborative_node,
    lab_analyst_collaborative_node,
    pharmacologist_collaborative_node,
)

# ==========================================================
# 1. ДИНАМІЧНИЙ КЛІНІЧНИЙ РОУТЕР (АРБІТРАЖ ЛАБОРАНТА)
# ==========================================================


def route_after_lab(state: MedicalSystemState) -> str:
    """
    Динамічно аналізує репліку Лаборанта на наявність зауважень до діагноста.
    """
    if state.discussion_turns >= 6:
        print("[Роутер] Досягнуто ліміту дискусії. Переходимо до призначення ліків.")
        return "pharmacologist"

    lab_messages = [msg for msg in state.messages if msg.agent_name == "Лаборант"]
    if not lab_messages:
        return "pharmacologist"

    last_verdict = lab_messages[-1].verdict.lower()
    print(f"[Роутер] Аналізуємо репліку Лаборанта: '{last_verdict}'")

    if "[маршрут: потрібен_перегляд]" in last_verdict:
        print(f"[Консенсус] ⚠️ Лаборант виявив помилку! Повертаємо справу Діагносту...")
        return "diagnostician"

    print("[Консенсус] ✅ Лаборант погодив картину. Передаємо дані Фармацевту.")
    return "pharmacologist"


# ==========================================================
# 2. ПОБУДОВА ДИНАМІЧНОГО ГРАФА КОНСЕНСУСУ
# ==========================================================

workflow = StateGraph(MedicalSystemState)

workflow.add_node("registrar", registrar_node)
workflow.add_node("diagnostician", diagnostician_collaborative_node)
workflow.add_node("lab_analyst", lab_analyst_collaborative_node)
workflow.add_node("pharmacologist", pharmacologist_collaborative_node)

workflow.add_edge(START, "registrar")
workflow.add_edge("registrar", "diagnostician")
workflow.add_edge("diagnostician", "lab_analyst")

workflow.add_conditional_edges(
    "lab_analyst",
    route_after_lab,
    {
        "diagnostician": "diagnostician",
        "pharmacologist": "pharmacologist",
    },
)

workflow.add_edge("pharmacologist", END)

collaborative_app = workflow.compile()
