import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelCriticidad

BASE = {
    "tecnologias": ["FastAPI"],
    "nivel_de_criticidad": "media",
    "resumen_tecnico": "API con degradación en horas pico.",
}


def test_valid_extraction():
    result = ExtraccionTecnica.model_validate(BASE)
    assert result.nivel_de_criticidad is NivelCriticidad.MEDIA


def test_empty_technologies_are_rejected():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(BASE | {"tecnologias": []})


def test_blank_technologies_are_rejected():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(BASE | {"tecnologias": ["  "]})


def test_duplicates_are_removed():
    result = ExtraccionTecnica.model_validate(BASE | {"tecnologias": ["Redis", " redis ", "Kafka"]})
    assert result.tecnologias == ["Redis", "Kafka"]


def test_unknown_criticality_is_rejected():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(BASE | {"nivel_de_criticidad": "urgente"})
