import pytest
from pydantic import SecretStr, ValidationError

from config import Settings
from schemas import ChatMessage, ChatRequest, GenerationParams, Provider, Role


@pytest.mark.parametrize("temperature", [-0.1, 2.1, 3.5])
def test_temperature_out_of_range(temperature):
    with pytest.raises(ValidationError):
        GenerationParams(temperature=temperature)


def test_invalid_max_tokens():
    with pytest.raises(ValidationError):
        GenerationParams(max_tokens=0)


def test_unknown_param_is_rejected():
    with pytest.raises(ValidationError):
        GenerationParams(top_k=5)


def test_empty_message_is_rejected():
    with pytest.raises(ValidationError):
        ChatMessage(role=Role.USER, content="")


@pytest.mark.parametrize(
    "roles",
    [
        [Role.SYSTEM],
        [Role.USER, Role.ASSISTANT],
    ],
)
def test_conversation_must_end_with_user(roles):
    messages = [ChatMessage(role=role, content="hola") for role in roles]
    with pytest.raises(ValidationError):
        ChatRequest(messages=messages)


def test_settings_require_api_key_of_selected_provider():
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(provider=Provider.ANTHROPIC, openai_api_key=SecretStr("sk-x"))


def test_settings_hide_api_key():
    settings = Settings(openai_api_key=SecretStr("sk-secreta"))
    assert "sk-secreta" not in repr(settings)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = Settings.from_env()

    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-5-5"
    assert settings.openai_api_key is None
