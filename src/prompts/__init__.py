from altair import value
from langchain_core.prompts import ChatPromptTemplate
from src.prompts import diagnostician, lab_analyst, pharmacologist, judge, orchestrator

"""
ГЛОБАЛЬНИЙ МЕНЕДЖЕР ШАБЛОНІВ ПРОМПТІВ
"""

diagnostician_blueprints = {
    "independent": diagnostician.INDEPENDENT,
    "collaborative": diagnostician.COLLABORATIVE,
    "centralized": diagnostician.CENTRALIZED,
}

lab_analyst_blueprints = {
    "independent": lab_analyst.INDEPENDENT,
    "collaborative": lab_analyst.COLLABORATIVE,
    "centralized": lab_analyst.CENTRALIZED,
}

pharmacologist_blueprints = {
    "independent": pharmacologist.INDEPENDENT,
    "collaborative": pharmacologist.COLLABORATIVE,
    "centralized": pharmacologist.CENTRALIZED,
}

DIAGNOSTICIAN_PROMPTS = {
    key: ChatPromptTemplate.from_messages(value)
    for key, value in diagnostician_blueprints.items()
}

LAB_ANALYST_PROMPTS = {
    key: ChatPromptTemplate.from_messages(value)
    for key, value in lab_analyst_blueprints.items()
}

PHARMACOLOGIST_PROMPTS = {
    key: ChatPromptTemplate.from_messages(value)
    for key, value in pharmacologist_blueprints.items()
}

JUDGE_PROMPT = judge.JUDGE_PROMPT
ORCHESTRATOR_PROMPT = orchestrator.ORCHESTRATOR_PROMPT
