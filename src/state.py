from typing import List, Optional, Annotated, Dict, Any
from pydantic import BaseModel, Field

# Нове поле
import operator


def merge_messages(left: list, right: list) -> list:
    """
    Функція-редюсер для об'єднання списків повідомлень агентів.
    """
    if not left:
        left = []
    if not right:
        return left
    new_messages = list(left)
    for msg in right:
        if msg not in new_messages:
            new_messages.append(msg)
    return new_messages


# ==========================================================
# БАЗОВІ МОДЕЛІ ДАНИХ ПАЦІЄНТА ТА МІЖАГЕНТНИХ ПОВІДОМЛЕНЬ
# ==========================================================


class AgentMessage(BaseModel):
    """Модель текстового вердикту від конкретного агента."""

    agent_name: str
    verdict: str


class PatientData(BaseModel):
    """Модель структурованих даних пацієнта."""

    name: str = Field(default="Невідомий пацієнт")
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(default="Не вказано")
    complaints: str = Field(...)
    metrics: Dict[str, Any] = Field(default_factory=dict)


# ==========================================================
# СТРУКТУРОВАНІ МОДЕЛІ ДЛЯ АВТОНОМНОГО ФАРМАЦЕВТА
# ==========================================================


class DrugSuggestion(BaseModel):
    """Опис рекомендованого лікарського засобу."""

    name: str = Field(..., description="Назва популярного бренду в Україні")
    active_ingredient: str = Field(..., description="Міжнародна діюча речовина (МНН)")
    target_symptom: str = Field(
        ..., description="Конкретний симптом пацієнта, для якого призначено ліки"
    )
    description: str = Field(..., description="Короткий опис фармакологічної дії")
    contraindications: str = Field(
        ..., description="Критичні ризики та протипоказання (шлунок, тиск, сумісність)"
    )
    administration: str = Field(
        ..., description="Рекомендований спосіб прийому та дозування"
    )


class PharmacologistOutput(BaseModel):
    """Контейнер для фінального списку фармакологічних призначень."""

    suggested_drugs: List[DrugSuggestion] = Field(
        default_factory=list, description="Список підібраних варіантів ліків"
    )


# ==========================================================
# МОДЕЛІ РІШЕНЬ ДЛЯ ЦЕНТРАЛІЗОВАНОГО ОРКЕСТРАТОРА
# ==========================================================


class OrchestratorDecision(BaseModel):
    """
    Валідаційна мапа фінального вердикту Головного лікаря.
    """

    has_errors: bool = Field(
        ...,
        description="True, якщо у звітах підлеглих є клінічні суперечності, помилки або ризики для пацієнта.",
    )
    diagnostician_feedback: Optional[str] = Field(
        default=None,
        description="Вказівка для Діагноста. Обов'язково None, якщо його діагноз ідеальний.",
    )
    lab_analyst_feedback: Optional[str] = Field(
        default=None,
        description="Вказівка для Лаборанта. Обов'язково None, якщо його аналіз ідеальний.",
    )
    pharmacologist_feedback: Optional[str] = Field(
        default=None,
        description="Вказівка для Фармацевта. Обов'язково None, якщо призначення на 100% безпечні.",
    )
    final_consolidated_report: Optional[str] = Field(
        default=None,
        description="Фінальний консолідований клінічний висновок. Генерується СУВОРO тоді, коли has_errors == False.",
    )


# ==========================================================
# ЗАГАЛЬНИЙ СТАН МЕДИЧНОЇ СИСТЕМИ (LANGGRAPH STATE)
# ==========================================================


class MedicalSystemState(BaseModel):
    """Єдиний операційний стан мультиагентної системи (LangGraph State)."""

    raw_text: str
    patient_data: Optional[PatientData] = None
    messages: Annotated[List[AgentMessage], merge_messages] = Field(
        default_factory=list
    )
    final_report: Optional[str] = None

    diagnostician_output: Optional[str] = None
    lab_analyst_output: Optional[str] = None
    pharmacologist_output: Optional[PharmacologistOutput] = None

    discussion_turns: int = Field(default=0)

    diagnostician_feedback: Optional[str] = None
    lab_analyst_feedback: Optional[str] = None
    pharmacologist_feedback: Optional[str] = None

    # НОВЕ ПОЛЕ: Накопичувальний лічильник токенів для всієї сесії
    total_tokens: Annotated[int, operator.add] = Field(default=0)
