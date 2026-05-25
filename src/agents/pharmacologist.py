from typing import Any, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import MedicalSystemState, AgentMessage, PharmacologistOutput
from src.prompts import PHARMACOLOGIST_PROMPTS

# ==========================================================
# ДОПОМІЖНІ ВНУТРІШНІ УТИЛІТИ
# ==========================================================


def _extract_text_content(content: Any) -> str:
    """
    Універсальний парсер відповідей від ChatGoogleGenerativeAI для текстових вузлів.
    """
    if isinstance(content, list):
        return " ".join(
            [part.get("text", "") for part in content if isinstance(part, dict)]
        )
    return str(content)


def format_pharmacologist_output(output: Optional[PharmacologistOutput]) -> str:
    """
    Публічна утиліта для конвертації структурованого каталогу ліків у читабельний текст.
    """
    if not output or not output.suggested_drugs:
        return "Каталог ліків порожній або відсутній."

    text_blocks = []
    for idx, drug in enumerate(output.suggested_drugs, 1):
        block = (
            f"ID-{idx}. Препарат: {drug.name}\n"
            f"   - Діюча речовина (МНН): {drug.active_ingredient}\n"
            f"   - Цільовий симптом: {drug.target_symptom}\n"
            f"   - Фармакологічна дія: {drug.description}\n"
            f"   - Протипоказання та ризики: {drug.contraindications}\n"
            f"   - Схема прийому: {drug.administration}"
        )
        text_blocks.append(block)

    return "\n\n".join(text_blocks)


def _format_structured_drugs_log(
    pydantic_output: PharmacologistOutput, is_centralized: bool = False
) -> str:
    """
    Генератор строкових звітів на основі структурованих Pydantic-моделей.
    """
    header = (
        "Оновлено список ліків за вказівкою Головного лікаря:\n"
        if is_centralized
        else "Згенеровано автономний список препаратів:\n"
    )
    log_text = header
    for drug in pydantic_output.suggested_drugs:
        log_text += (
            f"- {drug.name} ({drug.active_ingredient}) від '{drug.target_symptom}'\n"
        )
    return log_text


def _execute_pharmacologist_core(
    state: MedicalSystemState,
    prompt_key: str,
    extra_inputs: Optional[dict] = None,
    is_structured: bool = False,
) -> Tuple[Optional[PharmacologistOutput], Optional[str], int]:
    """
    Абстрактне ядро виконання для всіх топологій Агента-Фармацевта.
    Повертає: (pydantic_output, текстовий_вивід, витрачені_токени)
    """
    print("Агент-Фармацевт підбирає симптоматичне лікування...")

    patient = state.patient_data
    context_str = f"Пацієнт: {patient.age} років, стать: {patient.gender}.\nСкарги: {patient.complaints}\n"

    if patient.metrics:
        metrics_str = ", ".join([f"{k}: {v}" for k, v in patient.metrics.items()])
        context_str += f"Показники: {metrics_str}\n"

    inputs = {"context": context_str}
    if extra_inputs:
        inputs.update(extra_inputs)

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    if is_structured:
        # include_raw=True дозволяє зберегти метадані про токени разом із валідацією Pydantic
        structured_llm = llm.with_structured_output(
            PharmacologistOutput, include_raw=True
        )
        chain = PHARMACOLOGIST_PROMPTS[prompt_key] | structured_llm

        response_dict = chain.invoke(inputs)
        pydantic_output = response_dict["parsed"]  # Наш об'єкт структурованих ліків
        raw_message = response_dict["raw"]  # Сирий AIMessage для збору токенів

        usage = getattr(raw_message, "usage_metadata", {})
        tokens_spent = usage.get("total_tokens", 0)

        return pydantic_output, None, tokens_spent
    else:
        chain = PHARMACOLOGIST_PROMPTS[prompt_key] | llm
        response = chain.invoke(inputs)
        final_text = _extract_text_content(response.content)

        usage = getattr(response, "usage_metadata", {})
        tokens_spent = usage.get("total_tokens", 0)

        return None, final_text, tokens_spent


# ==========================================================
# ФУНКЦІЇ ВУЗЛІВ (NODES) ДЛЯ LANGGRAPH
# ==========================================================


def pharmacologist_independent_node(state: MedicalSystemState) -> dict:
    """
    Вузол Фармацевта в рамках Незалежної паралельної архітектурної топології.
    """
    pydantic_output, _, tokens = _execute_pharmacologist_core(
        state, "independent", is_structured=True
    )
    log_text = _format_structured_drugs_log(pydantic_output, is_centralized=False)

    new_message = AgentMessage(agent_name="Фармацевт", verdict=log_text)
    return {
        "pharmacologist_output": pydantic_output,
        "messages": [new_message],
        "total_tokens": tokens,  # Передаємо для підрахунку
    }


def pharmacologist_collaborative_node(state: MedicalSystemState) -> dict:
    """
    Вузол Фармацевта в рамках Децентралізованого колаборативного чату.
    """
    if state.messages:
        chat_history = "\n".join(
            [f"[{msg.agent_name}]: {msg.verdict}" for msg in state.messages]
        )
    else:
        chat_history = "Дискусія тільки розпочинається. Попередніх повідомлень немає."

    _, final_text, tokens = _execute_pharmacologist_core(
        state=state,
        prompt_key="collaborative",
        extra_inputs={"history": chat_history},
        is_structured=False,
    )

    new_message = AgentMessage(agent_name="Фармацевт", verdict=final_text)
    return {
        "messages": [new_message],
        "discussion_turns": state.discussion_turns + 1,
        "total_tokens": tokens,  # Передаємо для підрахунку
    }


def pharmacologist_centralized_node(state: MedicalSystemState) -> dict:
    """
    Вузол Фармацевта в рамках Centralized ієрархічної структури.
    """
    previous_verdict = ""
    if state.pharmacologist_output and state.pharmacologist_output.suggested_drugs:
        for idx, drug in enumerate(state.pharmacologist_output.suggested_drugs, 1):
            previous_verdict += f"{idx}. {drug.name} ({drug.active_ingredient}) — від симптому: '{drug.target_symptom}'\n"
    else:
        previous_verdict = "Каталог ліків ще не формувався (Первинний раунд)."

    feedback = (
        state.pharmacologist_feedback
        or "Зауважень немає. Сформуйте стартову карту ліків."
    )

    extra = {"previous_verdict": previous_verdict, "feedback": feedback}

    pydantic_output, _, tokens = _execute_pharmacologist_core(
        state=state, prompt_key="centralized", extra_inputs=extra, is_structured=True
    )
    log_text = _format_structured_drugs_log(pydantic_output, is_centralized=True)

    new_message = AgentMessage(agent_name="Фармацевт", verdict=log_text)
    return {
        "pharmacologist_output": pydantic_output,
        "messages": [new_message],
        "total_tokens": tokens,  # Передаємо для підрахунку
    }
