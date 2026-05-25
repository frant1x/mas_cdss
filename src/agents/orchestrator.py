from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import MedicalSystemState, AgentMessage, OrchestratorDecision
from src.prompts import ORCHESTRATOR_PROMPT
from src.agents import format_pharmacologist_output

"""
Вузол Головного Лікаря (Оркестратора), який збирає всі звіти та приймає рішення про безпеку призначень.
"""


def orchestrator_node(state: MedicalSystemState) -> dict:
    """
    Вузол Головного лікаря медичного центру.
    """
    print(
        f"\n[Раунд Оцінки {state.discussion_turns + 1}] Головний лікар (Оркестратор) проводить аудит звітів..."
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    # ДОДАНО: include_raw=True для збереження метаданих про токени в структурованому режимі
    structured_orchestrator = llm.with_structured_output(
        OrchestratorDecision, include_raw=True
    )

    p = state.patient_data
    patient_context = (
        f"Пацієнт: {p.name}, {p.age} років, стать: {p.gender}.\n"
        f"Первинні скарги: {p.complaints}\n"
        f"Об'єктивні метрики аналізів: {p.metrics}\n"
    )

    pharma_text = format_pharmacologist_output(state.pharmacologist_output)

    # ЗМІНЕНО: тепер invoke повертає словник із ключами 'parsed' та 'raw'
    response_dict = structured_orchestrator.invoke(
        ORCHESTRATOR_PROMPT.format(
            patient=patient_context,
            diag=state.diagnostician_output or "Відсутній",
            lab=state.lab_analyst_output or "Відсутній",
            pharma=pharma_text,
        )
    )

    decision = response_dict["parsed"]  # Наш Pydantic-об'єкт OrchestratorDecision
    raw_message = response_dict["raw"]  # Сира відповідь моделі для лічильника токенів

    # ВИТЯГУЄМО КІЛЬКІСТЬ ТОКЕНІВ ЗА ЦЕЙ КРОК
    usage = getattr(raw_message, "usage_metadata", {})
    tokens_spent = usage.get("total_tokens", 0)

    if decision.has_errors:
        log_text = (
            "🚨 Головний лікар виявив відхилення та повернув картку на доопрацювання:\n"
        )
        if decision.diagnostician_feedback:
            log_text += f"🔹 [До Діагноста]: {decision.diagnostician_feedback}\n"
        if decision.lab_analyst_feedback:
            log_text += f"🔹 [До Лаборанта]: {decision.lab_analyst_feedback}\n"
        if decision.pharmacologist_feedback:
            log_text += f"🔹 [До Фармацевта]: {decision.pharmacologist_feedback}\n"
    else:
        log_text = "✅ Головний лікар успешно затвердив усі звіти! Клінічний консенсус досягнуто."

    new_message = AgentMessage(
        agent_name="Головний Лікар (Оркестратор)", verdict=log_text
    )

    return {
        "diagnostician_feedback": decision.diagnostician_feedback,
        "lab_analyst_feedback": decision.lab_analyst_feedback,
        "pharmacologist_feedback": decision.pharmacologist_feedback,
        "final_report": decision.final_consolidated_report,
        "messages": [new_message],
        "discussion_turns": state.discussion_turns + 1,
        "total_tokens": tokens_spent,  # Передаємо редюсеру для автоматичного підсумовування
    }
