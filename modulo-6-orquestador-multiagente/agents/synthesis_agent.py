"""Agente sintetizador: redacta la respuesta final a partir de lo que aportaron los demás.
No tiene herramientas: solo puede usar los aportes que recibe."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.common import build_specialist_input, text_of
from state import Contribucion

SYSTEM_PROMPT = """Sos el redactor de un equipo. Escribís la respuesta final para el usuario \
usando únicamente los aportes del investigador y del analista.

- No agregues datos que no estén en los aportes. Si algo no está, decilo.
- Usá las cifras tal como las calculó el analista, sin recalcularlas.
- Si dos aportes se contradicen, mencioná la diferencia en vez de elegir uno en silencio.
- Respondé en español, con una comparación clara y una recomendación fundamentada. Citá \
qué agente aportó cada dato, por ejemplo "(analista)" o "(investigador)"."""


async def synthesize(
    model: BaseChatModel, question: str, instruction: str, contributions: list[Contribucion]
) -> str:
    prompt = build_specialist_input(question, instruction, contributions)
    response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
    return text_of(response)
