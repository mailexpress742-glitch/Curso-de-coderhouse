import pytest

from chain import PROMPT, build_chain, process_text
from fakes import VALID, FakeModel, good, malformed, truncated
from schemas import PipelineError


def make_chain(model: FakeModel, attempts: int = 3):
    return build_chain(model, max_attempts=attempts, jitter=False)


async def test_valid_output_needs_one_call():
    model = FakeModel(good())
    result = await process_text("texto", make_chain(model))
    assert result == VALID
    assert model.calls == 1


async def test_retries_after_malformed_output():
    model = FakeModel(malformed(), good())
    result = await process_text("texto", make_chain(model))
    assert result == VALID
    assert model.calls == 2


async def test_retries_after_truncated_output():
    model = FakeModel(truncated(), good())
    result = await process_text("texto", make_chain(model))
    assert result == VALID
    assert model.calls == 2


async def test_gives_up_after_max_attempts():
    model = FakeModel(malformed())
    with pytest.raises(PipelineError):
        await process_text("texto", make_chain(model, attempts=3))
    assert model.calls == 3


async def test_empty_text_is_rejected_without_calling_the_model():
    model = FakeModel(good())
    with pytest.raises(ValueError):
        await process_text("   ", make_chain(model))
    assert model.calls == 0


def test_prompt_takes_text_variable():
    assert PROMPT.input_variables == ["text"]
