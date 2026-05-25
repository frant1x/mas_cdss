from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import MedicalSystemState, PatientData, AgentMessage

"""
Універсальний вузол реєстрації пацієнта.
"""


def registrar_node(state: MedicalSystemState) -> dict:
    """
    Універсальний вузол реєстрації пацієнта.
    """
    print("[Запуск] Агент-Реєстратор структурує дані пацієнта...")

    if state.patient_data is not None:
        new_message = AgentMessage(
            agent_name="Реєстратор (Форма)",
            verdict="Клінічну карту успішно імпортовано напряму через цифрову форму UI.",
        )
        return {"messages": [new_message]}

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(PatientData)

    system_prompt = (
        "Ти — медичний реєстратор. Твоє завдання — уважно прочитати неструктурований текст від лікаря.\n"
        "Заповни ПІБ, вік, стать пацієнта (чоловіча/жіноча) та його скарги. Усі інші виявлені фізіологічні параметри, "
        "цифри аналізів (наприклад: тиск, цукор, пульс тощо) запиши у словник `metrics` у форматі 'назва_показника': 'значення'."
    )

    parsed_patient_data = structured_llm.invoke(
        [("system", system_prompt), ("user", state.raw_text)]
    )
    new_message = AgentMessage(
        agent_name="Реєстратор (Текст)",
        verdict=f"Успішно розпарсено сирий клінічний текст пацієнта {parsed_patient_data.name}. Структуровану карту створено за допомогою ШІ.",
    )
    return {"patient_data": parsed_patient_data, "messages": [new_message]}
