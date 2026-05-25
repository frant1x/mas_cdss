from typing import Any, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import MedicalSystemState, AgentMessage
from src.prompts import DIAGNOSTICIAN_PROMPTS

# ==========================================================
# ДОПОМІЖНІ ВНУТРІШНІ УТИЛІТИ
# ==========================================================


def _extract_text_content(content: Any) -> str:
    """
    Універсальний парсер відповідей від ChatGoogleGenerativeAI.
    """
    if isinstance(content, list):
        return " ".join(
            [part.get("text", "") for part in content if isinstance(part, dict)]
        )
    return str(content)


def _execute_diagnostician_core(
    state: MedicalSystemState, prompt_key: str, extra_inputs: Optional[dict] = None
) -> Tuple[str, AgentMessage]:
    """
    Абстрактне ядро виконання для всіх архітектур для Агента-Діагноста.
    """
    print("Агент-Діагност аналізує скарги пацієнта...")

    patient = state.patient_data
    context_str = f"Пацієнт: {patient.age} років, стать: {patient.gender}.\nСкарги: {patient.complaints}\n"

    inputs = {"context": context_str}
    if extra_inputs:
        inputs.update(extra_inputs)

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
    chain = DIAGNOSTICIAN_PROMPTS[prompt_key] | llm

    response = chain.invoke(inputs)
    final_text = _extract_text_content(response.content)

    # БЕЗПЕЧНО ВИДІЛЯЄМО МЕТАДАНІ ТОКЕНІВ ЗА КРОК
    usage = getattr(response, "usage_metadata", {})
    tokens_spent = usage.get("total_tokens", 0)

    return (
        final_text,
        AgentMessage(agent_name="Діагност", verdict=final_text),
        tokens_spent,  # Повертаємо кількість токенів для оновлення стану
    )


# ==========================================================
# ФУНКЦІЇ ВУЗЛІВ (NODES) ДЛЯ LANGGRAPH
# ==========================================================


def diagnostician_independent_node(state: MedicalSystemState) -> dict:
    """
    Вузол Діагноста в рамках незалежної архітектури.
    """
    final_text, new_message, tokens = _execute_diagnostician_core(state, "independent")
    return {
        "diagnostician_output": final_text,
        "messages": [new_message],
        "total_tokens": tokens,  # Передаємо редюсеру стану
    }


def diagnostician_collaborative_node(state: MedicalSystemState) -> dict:
    """
    Вузол Діагноста в рамках децентралізованої архітектури.
    """
    if state.messages:
        chat_history = "\n".join(
            [f"[{msg.agent_name}]: {msg.verdict}" for msg in state.messages]
        )
    else:
        chat_history = "Дискусія тільки розпочинається. Попередніх повідомлень немає."

    _, new_message, tokens = _execute_diagnostician_core(
        state=state, prompt_key="collaborative", extra_inputs={"history": chat_history}
    )
    return {
        "messages": [new_message],
        "discussion_turns": state.discussion_turns + 1,
        "total_tokens": tokens,  # Передаємо редюсеру стану
    }


def diagnostician_centralized_node(state: MedicalSystemState) -> dict:
    """
    Вузол Діагноста в рамках централізованої архітектури.
    """
    extra = {
        "previous_verdict": state.diagnostician_output
        or "Звіт ще не формувався (Первинний раунд).",
        "feedback": state.diagnostician_feedback
        or "Зауважень немає. Сформуйте стартову карту.",
    }

    final_text, new_message, tokens = _execute_diagnostician_core(
        state=state, prompt_key="centralized", extra_inputs=extra
    )
    return {
        "diagnostician_output": final_text,
        "messages": [new_message],
        "total_tokens": tokens,  # Передаємо редюсеру стану
    }
