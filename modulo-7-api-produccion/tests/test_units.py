import os

import fakeredis
import pytest

from app.agents.supervisor import Limits, SupervisorDecision, apply_guards
from app.config import Settings
from app.jobs import JobStatus, JobStore
from app.observability import setup_observability


def decide(siguiente, instruccion="x"):
    return SupervisorDecision(siguiente=siguiente, instruccion=instruccion, razon="porque sí")


def contribution(agent, attempt=1):
    return {"agente": agent, "intento": attempt, "instruccion": "x", "resultado": "y", "datos": []}


def settings(**values) -> Settings:
    return Settings(anthropic_api_key="x", provider="anthropic", **values)


@pytest.fixture
def store():
    return JobStore(fakeredis.FakeAsyncRedis(decode_responses=True))


async def test_job_is_created_pending(store):
    job = await store.create("una pregunta")
    assert job.status == JobStatus.PENDING
    assert (await store.get(job.id)).pregunta == "una pregunta"


async def test_update_changes_fields_and_clears_none_values(store):
    job = await store.create("una pregunta")
    await store.update(job.id, status=JobStatus.WAITING_APPROVAL, accion_pendiente={"a": 1})
    assert (await store.get(job.id)).accion_pendiente == {"a": 1}
    await store.update(job.id, accion_pendiente=None, status=JobStatus.DONE)
    got = await store.get(job.id)
    assert got.accion_pendiente is None
    assert got.status == JobStatus.DONE


async def test_transition_only_works_from_the_expected_state(store):
    job = await store.create("una pregunta")
    assert not await store.transition(job.id, JobStatus.WAITING_APPROVAL, JobStatus.PENDING)
    await store.update(job.id, status=JobStatus.WAITING_APPROVAL)
    assert await store.transition(job.id, JobStatus.WAITING_APPROVAL, JobStatus.PENDING)
    assert not await store.transition(job.id, JobStatus.WAITING_APPROVAL, JobStatus.PENDING)


async def test_events_keep_their_order(store):
    job = await store.create("una pregunta")
    for n in range(3):
        await store.append_event(job.id, {"paso": n})
    assert [e["paso"] for e in (await store.get(job.id)).flujo] == [0, 1, 2]


def test_the_action_agent_cannot_run_before_the_report():
    destination, _, adjustment = apply_guards(decide("ejecutor"), [], 1, Limits())
    assert destination == "sintetizador"
    assert "informe" in adjustment


def test_the_action_agent_runs_at_most_once():
    done = [contribution("sintetizador"), contribution("ejecutor")]
    destination, _, adjustment = apply_guards(decide("ejecutor"), done, 5, Limits())
    assert destination == "FINISH"
    assert "agotó" in adjustment


def test_the_action_agent_is_allowed_after_the_report():
    result = apply_guards(decide("ejecutor"), [contribution("sintetizador")], 4, Limits())
    assert result == ("ejecutor", "x", None)


def test_observability_is_off_without_an_endpoint():
    assert setup_observability(settings()) is False


def test_observability_registers_phoenix_when_there_is_an_endpoint(monkeypatch):
    import phoenix.otel
    from openinference.instrumentation.langchain import LangChainInstrumentor

    calls = {}
    instrumented = []
    provider = object()

    def fake_register(**kwargs):
        calls.update(kwargs)
        return provider

    monkeypatch.setattr(phoenix.otel, "register", fake_register)
    monkeypatch.setattr(
        LangChainInstrumentor, "instrument", lambda self, **kw: instrumented.append(kw)
    )
    enabled = setup_observability(
        settings(phoenix_endpoint="http://localhost:6006/v1/traces", phoenix_project="demo")
    )
    assert enabled is True
    assert calls["endpoint"] == "http://localhost:6006/v1/traces"
    assert calls["project_name"] == "demo"
    # Solo se instrumenta LangChain (no se trazan los requests HTTP de FastAPI).
    assert "auto_instrument" not in calls
    assert instrumented == [{"tracer_provider": provider}]


def test_observability_turns_off_the_anthropic_sdk_spans_to_avoid_double_counting(monkeypatch):
    import phoenix.otel
    from openinference.instrumentation.langchain import LangChainInstrumentor

    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    monkeypatch.setattr(phoenix.otel, "register", lambda **kwargs: object())
    monkeypatch.setattr(LangChainInstrumentor, "instrument", lambda self, **kw: None)
    setup_observability(settings(phoenix_endpoint="http://localhost:6006/v1/traces"))
    assert os.environ["ANTHROPIC_OPEN_TELEMETRY"] == "false"
    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY")


def test_settings_require_the_api_key_of_the_provider():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(provider="openai")
