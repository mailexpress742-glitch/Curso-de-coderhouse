from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Contrato de salida del pipeline. El modelo tiene que completar estos campos."""

    tecnologias: list[str] = Field(
        min_length=1,
        description="Tecnologías, lenguajes, frameworks o servicios mencionados en el texto.",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Qué tan urgente es atender lo que describe el texto.",
    )
    resumen_tecnico: str = Field(
        min_length=10,
        max_length=400,
        description="Resumen técnico de una o dos oraciones.",
    )

    @field_validator("tecnologias")
    @classmethod
    def clean_technologies(cls, values: list[str]) -> list[str]:
        # Saca espacios y repetidos (sin importar mayúsculas) manteniendo el orden.
        seen: dict[str, str] = {}
        for value in values:
            name = value.strip()
            if name:
                seen.setdefault(name.lower(), name)
        if not seen:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return list(seen.values())


class PipelineError(Exception):
    """El pipeline no pudo devolver una extracción válida después de los reintentos."""


class IncompleteOutputError(Exception):
    """El modelo cortó la respuesta (por ejemplo, por falta de tokens) o no devolvió el esquema."""
