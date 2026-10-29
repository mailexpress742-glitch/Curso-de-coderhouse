---
categoria: validacion
etiquetas: Field, field_validator, model_validator
---

# Validadores y restricciones en Pydantic v2

## Restricciones con Field

La función `Field` agrega restricciones a un campo sin escribir código de validación. Para números se usan `gt`, `ge`, `lt` y `le` (mayor que, mayor o igual, menor que, menor o igual). Para strings y listas se usan `min_length` y `max_length`. Para strings también existe `pattern`, que recibe una expresión regular.

```python
from pydantic import BaseModel, Field


class Pedido(BaseModel):
    cantidad: int = Field(ge=1, le=100)
    codigo: str = Field(pattern=r"^[A-Z]{3}-\d{4}$")
    notas: str = Field(default="", max_length=200)
```

`Field` también acepta `description` y `examples`, que se incluyen en el esquema JSON generado y sirven para documentar una API.

## field_validator

Cuando una restricción no se puede expresar con `Field`, se escribe un validador con el decorador `@field_validator`. Recibe el nombre del campo y se declara como método de clase. El parámetro `mode` indica cuándo corre: con `mode="after"` (el valor por defecto) corre después de que Pydantic validó el tipo, y con `mode="before"` corre antes, sobre el dato crudo, lo que sirve para normalizarlo.

```python
from pydantic import BaseModel, field_validator


class Cuenta(BaseModel):
    usuario: str

    @field_validator("usuario")
    @classmethod
    def sin_espacios(cls, valor: str) -> str:
        if " " in valor:
            raise ValueError("el usuario no puede tener espacios")
        return valor.lower()
```

Un validador debe devolver el valor, que puede estar modificado, o lanzar `ValueError` o `AssertionError`. Pydantic convierte esa excepción en un error de validación con la ubicación del campo.

## model_validator

Para reglas que involucran varios campos a la vez se usa `@model_validator`. Con `mode="after"` recibe la instancia ya construida, y con `mode="before"` recibe los datos crudos.

```python
from typing import Self
from pydantic import BaseModel, model_validator


class Rango(BaseModel):
    minimo: int
    maximo: int

    @model_validator(mode="after")
    def minimo_menor_que_maximo(self) -> Self:
        if self.minimo > self.maximo:
            raise ValueError("el mínimo no puede superar al máximo")
        return self
```

## Annotated y validadores reutilizables

La forma recomendada de reutilizar una validación en varios campos es `typing.Annotated` junto con los marcadores `AfterValidator` y `BeforeValidator`. Se define el tipo una vez y se usa en todos los modelos.

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel


def a_minusculas(valor: str) -> str:
    return valor.strip().lower()


Etiqueta = Annotated[str, AfterValidator(a_minusculas)]


class Articulo(BaseModel):
    etiquetas: list[Etiqueta]
```

Con `Annotated` también se combinan restricciones de `Field`, por ejemplo `Annotated[int, Field(gt=0)]`, para crear alias de tipo como `EnteroPositivo`.

## Campos calculados

El decorador `@computed_field` agrega una propiedad a la serialización del modelo. La propiedad se calcula a partir de otros campos y aparece al llamar a `model_dump`, pero no se valida al crear la instancia, porque no es una entrada.

## Orden de ejecución

Cuando un campo tiene varios validadores, los de modo `before` corren de derecha a izquierda, después corre la validación del tipo, y los de modo `after` corren de izquierda a derecha. Los validadores de modelo corren al final, cuando todos los campos ya se validaron.
