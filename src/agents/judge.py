from src.state import MedicalSystemState, AgentMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompts import JUDGE_PROMPT
from src.agents import format_pharmacologist_output

"""
Вузол Клінічного Арбітра консиліуму.
"""


def judge_node(state: MedicalSystemState) -> dict:
    """
    Вузол Клінічного Арбітра консиліуму.
    """
    print("\nАгент-Суддя формує фінальний консолідований звіт...")
    llm_judge = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    pharma_text = format_pharmacologist_output(state.pharmacologist_output)

    p = state.patient_data
    patient_context = (
        f"ПІБ: {p.name if p else 'Не вказано'}\n"
        f"Вік: {p.age if p else 'Не вказано'} років\n"
        f"Стать: {p.gender if p else 'Не вказано'}\n"
        f"Анамнез скарг: {p.complaints if p else 'Не вказано'}\n"
    )

    chain = JUDGE_PROMPT | llm_judge
    response = chain.invoke(
        {
            "patient": patient_context,
            "diag": state.diagnostician_output or "Немає даних",
            "lab": state.lab_analyst_output or "Немає даних",
            "pharma": pharma_text,
        }
    )

    final_text = response.content
    if isinstance(final_text, list):
        final_text = " ".join(
            [part.get("text", "") for part in final_text if isinstance(part, dict)]
        )

    # ЗЧИТУЄМО МЕТАДАНІ ТОКЕНІВ ДЛЯ СУДДІ
    usage = getattr(response, "usage_metadata", {})
    tokens_spent = usage.get("total_tokens", 0)

    new_message = AgentMessage(agent_name="Консиліум (Суддя)", verdict=final_text)

    return {
        "final_report": final_text,
        "messages": [new_message],
        "total_tokens": tokens_spent,  # Плюсуємо токени в стан графа
    }
