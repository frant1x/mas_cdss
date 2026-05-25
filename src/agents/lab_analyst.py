from typing import Any, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import MedicalSystemState, AgentMessage
from src.prompts import LAB_ANALYST_PROMPTS

# ==========================================================
# ДОПОМІЖНІ ВНУТРІШНІ УТИЛІТИ
# ==========================================================


def _extract_text_content(content: Any) -> str:
    """
    Універсальний парсер відповідей від ChatGoogleGenerativeAI.

    Забезпечує безпечне вилучення рядкового контенту у випадках,
    коли модель повертає дані у вигляді списку фрагментів.
    """
    if isinstance(content, list):
        return " ".join(
            [part.get("text", "") for part in content if isinstance(part, dict)]
        )
    return str(content)


def _execute_lab_analyst_core(
    state: MedicalSystemState, prompt_key: str, extra_inputs: Optional[dict] = None
) -> Tuple[str, AgentMessage]:
    """
    Абстрактне ядро виконання для всіх топологій Агента-Лаборанта.

    Логує початок аналізу, автоматично витягує біометрію пацієнта,
    трансформує словник метрик у текстовий список та викликає ШІ-ланцюжок.
    """
    print("Агент-Лаборант аналізує метрики пацієнта...")

    patient = state.patient_data
    age = patient.age

    metrics_text = "\n".join(
        [f"- {key}: {value}" for key, value in patient.metrics.items()]
    )
    if not metrics_text:
        metrics_text = "Жодних цифрових чи лабораторних показників не виявлено."

    inputs = {"age": age, "metrics_text": metrics_text}
    if extra_inputs:
        inputs.update(extra_inputs)

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0)
    chain = LAB_ANALYST_PROMPTS[prompt_key] | llm

    response = chain.invoke(inputs)
    final_text = _extract_text_content(response.content)

    # ВИТЯГУЄМО ТОКЕНИ З МЕТАДАНИХ ВІДПОВІДІ
    usage = getattr(response, "usage_metadata", {})
    tokens_spent = usage.get("total_tokens", 0)

    return (
        final_text,
        AgentMessage(agent_name="Лаборант", verdict=final_text),
        tokens_spent,
    )


# ==========================================================
# ФУНКЦІЇ ВУЗЛІВ (NODES) ДЛЯ LANGGRAPH
# ==========================================================


def lab_analyst_independent_node(state: MedicalSystemState) -> dict:
    """
    Вузол Лаборанта в рамках Незалежної паралельної архітектурної топології.
    """
    final_text, new_message, tokens = _execute_lab_analyst_core(state, "independent")
    return {
        "lab_analyst_output": final_text,
        "messages": [new_message],
        "total_tokens": tokens,  # Плюсуємо токени в стан графа
    }


def lab_analyst_collaborative_node(state: MedicalSystemState) -> dict:
    """
    Вузол Лаборанта в рамках Децентралізованого колаборативного чату.
    """
    if state.messages:
        chat_history = "\n".join(
            [f"[{msg.agent_name}]: {msg.verdict}" for msg in state.messages]
        )
    else:
        chat_history = "Дискусія тільки розпочинається. Попередніх повідомлень немає."

    _, new_message, tokens = _execute_lab_analyst_core(
        state=state, prompt_key="collaborative", extra_inputs={"history": chat_history}
    )
    return {
        "messages": [new_message],
        "discussion_turns": state.discussion_turns + 1,
        "total_tokens": tokens,  # Плюсуємо токени в стан графа
    }


def lab_analyst_centralized_node(state: MedicalSystemState) -> dict:
    """
    Вузол Лаборанта в рамках Централізованої ієрархічної структури.
    """
    extra = {
        "previous_verdict": state.lab_analyst_output
        or "Звіт ще не формувався (Первинний раунд).",
        "feedback": state.lab_analyst_feedback
        or "Зауважень немає. Проведіть базовий аналіз наданих метрик пацієнта.",
    }

    final_text, new_message, tokens = _execute_lab_analyst_core(
        state=state, prompt_key="centralized", extra_inputs=extra
    )
    return {
        "lab_analyst_output": final_text,
        "messages": [new_message],
        "total_tokens": tokens,  # Плюсуємо токени в стан графа
    }
