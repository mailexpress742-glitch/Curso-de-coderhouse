---
categoria: validacion
etiquetas: validate_call, dataclasses, TypedDict
---

# Validar funciones, dataclasses y TypedDict

## El decorador validate_call

No siempre hace falta un modelo. El decorador `@validate_call` valida los argumentos de una función según sus anotaciones de tipo, cada vez que se la llama. Convierte los valores cuando es posible, como pasar `"3"` a `3`, y lanza una `ValidationError` si no puede.

```python
from pydantic import validate_call, Field
from typing import Annotated


@validate_call
def repetir(texto: str, veces: Annotated[int, Field(ge=1, le=10)] = 1) -> str:
    return texto * veces
```

Con el argumento `validate_return=True` también se valida el valor que devuelve la función. El decorador acepta un parámetro `config` con un `ConfigDict` para ajustar el comportamiento, por ejemplo activar el modo strict. Es una buena herramienta para validar funciones que son el punto de entrada de una librería, como las herramientas (tools) que un agente de IA puede invocar.

## Dataclasses de Pydantic

Las dataclasses de la biblioteca estándar no validan nada: aceptan cualquier valor. Pydantic ofrece `pydantic.dataclasses.dataclass`, un decorador que agrega validación a una dataclass manteniendo su sintaxis. Funciona de forma parecida a un `BaseModel`, pero el resultado sigue siendo una dataclass normal, compatible con herramientas que esperan ese tipo.

```python
from pydantic.dataclasses import dataclass


@dataclass
class Punto:
    x: int
    y: int
```

También se puede usar una dataclass estándar como campo de un `BaseModel`: Pydantic la valida al validar el modelo contenedor, sin necesidad de decorarla.

## BaseModel o dataclass

Las diferencias prácticas son pocas. `BaseModel` ofrece más funcionalidad: `model_dump`, `model_json_schema`, configuración detallada y herencia con campos privados. Las dataclasses de Pydantic son preferibles cuando se necesita compatibilidad con código existente que usa `dataclasses.fields` o `dataclasses.asdict`. Para una API nueva, `BaseModel` es la opción por defecto.

## TypedDict

Un `TypedDict` describe la forma de un diccionario con claves y tipos conocidos. Pydantic puede validar diccionarios contra un `TypedDict` usando `TypeAdapter`. Hay un detalle importante: en versiones de Python anteriores a la 3.12 hay que importar `TypedDict` desde `typing_extensions` y no desde `typing`, porque Pydantic necesita la versión extendida para leer las anotaciones completas.

```python
from typing_extensions import TypedDict
from pydantic import TypeAdapter


class Libro(TypedDict):
    titulo: str
    paginas: int


TypeAdapter(Libro).validate_python({"titulo": "Dune", "paginas": "412"})
```

El resultado es un diccionario normal, con `paginas` convertido a entero. Un `TypedDict` no crea una instancia de una clase: no tiene métodos ni atributos, solo valida la estructura.

## NamedTuple y otros tipos

Pydantic también valida `NamedTuple`, conjuntos (`set`), tuplas con tipos por posición (`tuple[int, str]`), diccionarios con claves tipadas (`dict[str, float]`) y secuencias. En cada caso se usa como anotación de un campo o con `TypeAdapter`.

## Validación de parámetros en agentes

En sistemas con modelos de lenguaje es habitual validar los argumentos que el modelo propone para una herramienta antes de ejecutarla. Un `BaseModel` que describe los parámetros cumple dos funciones a la vez: su `model_json_schema()` se envía al modelo como descripción de la herramienta, y `model_validate` valida los argumentos que el modelo devuelve.
