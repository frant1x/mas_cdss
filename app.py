from dotenv import load_dotenv
import streamlit as st

from src.architectures.independent import independent_app
from src.architectures.collaborative import collaborative_app
from src.architectures.centralized import centralized_app
from src.state import PatientData

# ==========================================================
# 1. НАЛАШТУВАННЯ СТОРІНКИ ТА СЕСІЇ
# ==========================================================

load_dotenv()

st.set_page_config(
    page_title="МАС Консиліум",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

AGENT_AVATARS = {
    "Реєстратор": "📋",
    "Діагност": "💡",
    "Лаборант": "🔬",
    "Фармацевт": "💊",
    "Оркестратор": "👑",
    "Головний Лікар": "👑",
}

# ==========================================================
# 2. UI: БІЧНА ПАНЕЛЬ (SIDEBAR КОНФІГУРАЦІЯ)
# ==========================================================

with st.sidebar:
    st.header("⚙️ Керування системою")
    st.markdown("---")
    st.markdown("### 🧩 Доступні архітектури МАС:")

    topology_mode = st.radio(
        "Вибір топології",
        [
            "🟢 Незалежні агенти (Independent)",
            "🔵 Круглий стіл (Collaborative)",
            "🟣 Централізований контроль (Orchestrator)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

# ==========================================================
# 3. UI: ГОЛОВНИЙ РОБОЧИЙ ПРОСТІР
# ==========================================================

st.title("🩺 СППКР: Мультиагентний клінічний консиліум")
st.subheader("Гнучке введення клінічних даних пацієнта")

input_mode = st.radio(
    "🔮 Оберіть зручний формат введення клінічних даних пацієнта:",
    [
        "📝 Вільний неструктурований текст (Аналіз ШІ)",
        "📋 Цифрова картка пацієнта (Ручне заповнення форми)",
    ],
    horizontal=True,
)

st.markdown("---")

# ==========================================================
# 4. ФОРМУВАННЯ ВХІДНОГО СТАНУ (INITIAL STATE)
# ==========================================================

if input_mode == "📝 Вільний неструктурований текст (Аналіз ШІ)":
    default_text = "До клініки звернулася пацієнтка Олійник Наталія Вікторівна, 70 років. Скаржиться на гострий, пекучий біль у колінних та кульшових суглобах, який заважає ходити. Загострення остеоартриту відбулося після тривалої прогулянки. Також скаржиться на періодичний ниючий біль у епігастрії. При огляді зафіксовано: температура тіла 36.8 °C, артеріальний тиск 140/90 мм рт. ст., пульс 76 уд/хв. Через миготливу аритмію та високий ризик інсульту пацієнтка щоденно приймає сильний антикоагулянт Ксарелто (ривароксабан)."
    raw_input = st.text_area(
        "Введіть сирий текст огляду чи анамнезу пацієнта:",
        value=default_text,
        height=150,
    )

    initial_state = {
        "raw_text": raw_input,
        "patient_data": None,
        "messages": [],
        "discussion_turns": 0,
    }

else:
    st.markdown("### 📋 Заповнення медичної карти пацієнта")
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

    with f_col1:
        p_name = st.text_input("ПІБ пацієнта:", value="Петренко Анна Сергіївна")
    with f_col2:
        p_age = st.number_input(
            "Вік (повних років):", min_value=1, max_value=120, value=24
        )
    with f_col3:
        p_gender = st.selectbox("Стать пацієнта:", ["Жіноча", "Чоловіча", "Не вказано"])

    p_complaints = st.text_area(
        "Основні скарги пацієнта:",
        value="Сильний біль у горлі при ковтанні, закладеність носа, сухий кашель, загальна слабкість, озноб.",
    )

    st.markdown("**📊 Об'єктивні показники огляду:**")
    st.caption(
        "💡 Ви можете змінювати назви показників, редагувати значення, видаляти рядки або додавати нові."
    )

    default_metrics = [
        {"Показник": "Температура тіла", "Значення": "37.8 °C"},
        {"Показник": "Пульс", "Значення": "82 уд/хв"},
        {"Показник": "Хронічні захворювання", "Значення": "легкий гастрит"},
    ]

    edited_metrics_table = st.data_editor(
        default_metrics,
        num_rows="dynamic",
        column_config={
            "Показник": st.column_config.TextColumn("Назва показника", width="medium"),
            "Значення": st.column_config.TextColumn(
                "Клінічне значення", width="medium"
            ),
        },
        width="stretch",
    )

    metrics_dict = {
        row["Показник"].strip(): row["Значення"].strip()
        for row in edited_metrics_table
        if row.get("Показник") and row.get("Значення")
    }

    created_patient = PatientData(
        name=p_name,
        age=p_age,
        gender=p_gender,
        complaints=p_complaints,
        metrics=metrics_dict,
    )

    initial_state = {
        "raw_text": "Клінічні дані завантажено безпосередньо через цифрову форму UI.",
        "patient_data": created_patient,
        "messages": [],
        "discussion_turns": 0,
    }

# ==========================================================
# 5. 🚀 ЗАПУСК КОНСИЛІУМУ ТА ЖИВИЙ СТРІМІНГ ГРАФА
# ==========================================================

if st.button("🚀 Запустити консиліум агентів", type="primary"):
    if (
        input_mode == "📝 Вільний неструктурований текст (Аналіз ШІ)"
        and not raw_input.strip()
    ):
        st.error("Будь ласка, введіть текст анамнезу.")
    else:
        col_left, col_right = st.columns([1, 2])

        # --------------------------------------------------
        # ЛІВА КОЛОНКА: Статично-динамічна панель пацієнта
        # --------------------------------------------------
        with col_left:
            st.header("📋 Дані пацієнта")
            patient_placeholder = st.empty()

            if initial_state["patient_data"] is not None:
                p = initial_state["patient_data"]
                with patient_placeholder.container():
                    st.metric(label="👤 ПІБ пацієнта", value=p.name)
                    p_col1, p_col2 = st.columns(2)
                    p_col1.metric(label="🎂 Вік", value=f"{p.age} р.")
                    p_col2.metric(label="🧬 Стать", value=p.gender)
                    st.markdown("**📝 Скарги:**")
                    st.info(p.complaints)
                    if p.metrics:
                        st.markdown("**📊 Виділені клінічні показники:**")
                        for key, value in p.metrics.items():
                            st.markdown(f"🔹 **{key.strip().capitalize()}:** {value}")
            else:
                patient_placeholder.caption(
                    "⏳ Очікування аналізу тексту та реєстрації пацієнта..."
                )

        # --------------------------------------------------
        # ПРАВА КОЛОНКА: Динамічні архітектурні контейнери
        # --------------------------------------------------
        with col_right:
            if topology_mode == "🟢 Незалежні агенти (Independent)":
                st.header("🔍 Експертні висновки консиліуму")
                tab_diag, tab_lab, tab_pharma = st.tabs(
                    ["💡 Агент-Діагност", "🔬 Агент-Лаборант", "💊 Агент-Фармацевт"]
                )

                with tab_diag:
                    diag_placeholder = st.empty()
                    diag_placeholder.caption("⏳ Очікування аналізу скарг...")
                with tab_lab:
                    lab_placeholder = st.empty()
                    lab_placeholder.caption(
                        "⏳ Очікування аналізу лабораторних показників..."
                    )
                with tab_pharma:
                    pharma_placeholder = st.empty()
                    pharma_placeholder.caption(
                        "⏳ Очікування формування каталогу ліків..."
                    )

                st.markdown("---")
                st.header("🏥 Фінальний вердикт голови консиліуму (Суддя)")
                judge_placeholder = st.empty()
                judge_placeholder.caption(
                    "⏳ Очікування фінального узгодження рішень експертів..."
                )

            elif topology_mode == "🔵 Круглий стіл (Collaborative)":
                st.header("💬 Жива дискусія консиліуму (Круглий стіл)")
                chat_container = st.container()

            else:
                st.header("📜 Командний журнал Головного лікаря (Оркестратор)")
                central_log_container = st.container()
                st.markdown("---")
                st.header("🏥 Затверджений висновок Головного лікаря")
                central_final_placeholder = st.empty()
                central_final_placeholder.caption(
                    "⏳ Очікування фінального ітеративного аудиту..."
                )

        # --------------------------------------------------
        # ЕКЗЕКУЦІЯ ТА СТРІМІНГ ОНОВЛЕНЬ З LANGGRAPH
        # --------------------------------------------------
        try:
            if topology_mode == "🟢 Незалежні агенти (Independent)":
                active_app = independent_app
            elif topology_mode == "🔵 Круглий стіл (Collaborative)":
                active_app = collaborative_app
            else:
                active_app = centralized_app

            # 🛠️ ІНІЦІАЛІЗАЦІЯ ЛІЧИЛЬНИКІВ ДЛЯ КОНСОЛІ
            tokens_accumulator = 0
            turns_accumulator = 0

            for chunk in active_app.stream(initial_state, stream_mode="updates"):
                for node_name, response in chunk.items():

                    # 🛠️ ЗБИРАЄМО МЕТРИКИ З КОЖНОГО ЧАНКА ОНОВЛЕНЬ
                    if isinstance(response, dict):
                        if "total_tokens" in response:
                            tokens_accumulator += response["total_tokens"]
                        if "discussion_turns" in response:
                            turns_accumulator = response["discussion_turns"]

                    # 1. Синхронна перемальовка картки пацієнта після роботи універсального Реєстратора
                    if (
                        node_name == "registrar"
                        and input_mode
                        == "📝 Вільний неструктурований текст (Аналіз ШІ)"
                    ):
                        patient = response.get("patient_data")
                        if patient:
                            with patient_placeholder.container():
                                st.metric(label="👤 ПІБ пацієнта", value=patient.name)
                                p_col1, p_col2 = st.columns(2)
                                p_col1.metric(label="🎂 Вік", value=f"{patient.age} р.")
                                p_col2.metric(label="🧬 Стать", value=patient.gender)
                                st.markdown("**📝 Скарги:**")
                                st.info(patient.complaints)
                                if patient.metrics:
                                    st.markdown("**📊 Виділені клінічні показники:**")
                                    for key, value in patient.metrics.items():
                                        st.markdown(
                                            f"🔹 **{key.strip().capitalize()}:** {value}"
                                        )

                    # 2. Логіка виводу для НЕЗАЛЕЖНОЇ паралельної схеми
                    if topology_mode == "🟢 Незалежні агенти (Independent)":
                        if node_name == "diagnostician":
                            diag_placeholder.write(response.get("diagnostician_output"))
                        elif node_name == "lab_analyst":
                            lab_placeholder.write(response.get("lab_analyst_output"))
                        elif node_name == "pharmacologist":
                            pharma_output = response.get("pharmacologist_output")
                            if pharma_output and pharma_output.suggested_drugs:
                                with pharma_placeholder.container():
                                    for drug in pharma_output.suggested_drugs:
                                        with st.expander(
                                            f"📦 {drug.name} ({drug.active_ingredient})"
                                        ):
                                            st.markdown(
                                                f"**🎯 Цільовий симптом:** {drug.target_symptom}"
                                            )
                                            st.markdown(
                                                f"**ℹ️ Дія:** {drug.description}"
                                            )
                                            st.markdown(
                                                f"**⚠️ Протипоказання:** {drug.contraindications}"
                                            )
                                            st.markdown(
                                                f"**⏰ Дозування:** {drug.administration}"
                                            )
                        elif node_name == "judge":
                            judge_placeholder.markdown(response.get("final_report"))
                            st.toast(
                                "Паралельний консиліум успішно завершено!", icon="✅"
                            )

                    # 3. Логіка виводу для ДЕЦЕНТРАЛІЗОВАНОГО ЧАТУ (Круглий стіл)
                    elif topology_mode == "🔵 Круглий стіл (Collaborative)":
                        if "messages" in response:
                            for msg in response["messages"]:
                                avatar = next(
                                    (
                                        icon
                                        for name, icon in AGENT_AVATARS.items()
                                        if name in msg.agent_name
                                    ),
                                    "🩺",
                                )
                                with chat_container.chat_message(
                                    msg.agent_name, avatar=avatar
                                ):
                                    st.markdown(f"**{msg.agent_name}**")
                                    st.write(msg.verdict)

                    # 4. Логіка виводу для ЦЕНТРАЛІЗОВАНОГО ОРКЕСТРАТОРА (Головний лікар)
                    else:
                        if "messages" in response:
                            for msg in response["messages"]:
                                avatar = next(
                                    (
                                        icon
                                        for name, icon in AGENT_AVATARS.items()
                                        if name in msg.agent_name
                                    ),
                                    "🩺",
                                )
                                with central_log_container.chat_message(
                                    msg.agent_name, avatar=avatar
                                ):
                                    st.markdown(f"**{msg.agent_name}**")
                                    st.write(msg.verdict)

                        if node_name == "orchestrator" and response.get("final_report"):
                            central_final_placeholder.markdown(
                                response.get("final_report")
                            )
                            st.toast(
                                "Ітеративний аудит завершено! Звіт підписано.",
                                icon="👑",
                            )

            if topology_mode == "🔵 Круглий стіл (Collaborative)":
                st.toast(
                    "Консиліум успішно прийшов до спільного консенсусу!", icon="🤝"
                )

            # 🚀 ВИВІД МЕТРИК У ТЕРМІНАЛ ПІСЛЯ ЗАВЕРШЕННЯ СТРІМІНГУ
            print("\n" + "=" * 60)
            print(f"📊 [ЕКСПЕРИМЕНТАЛЬНІ МЕТРИКИ СЕСІЇ]")
            print(f"🔹 Обрана архітектура: {topology_mode}")
            print(f"🔹 Сумарно витрачено токенів: {tokens_accumulator}")
            print(f"🔹 Фінальна кількість раундів: {turns_accumulator}")
            print("=" * 60 + "\n")

        except Exception as e:
            st.error(f"❌ Помилка під час виконання стрімінгу граф-системи: {e}")
