---
categoria: errores
etiquetas: ValidationError, PydanticCustomError, errors, loc
---

# Manejo de errores de validación

## La estructura de ValidationError

Cuando una validación falla, Pydantic lanza una `ValidationError`. No es una excepción simple con un mensaje: agrupa todos los errores encontrados en los datos. Un mismo modelo puede devolver cinco errores a la vez, uno por cada campo inválido, lo que permite mostrar al usuario todo lo que debe corregir en una sola respuesta.

El método `errors()` devuelve una lista de diccionarios. Cada elemento tiene estas claves:

- `type`: un identificador estable del error, como `missing`, `int_parsing` o `string_too_short`.
- `loc`: una tupla con la ubicación del error. Para un campo anidado incluye la ruta completa, por ejemplo `("direccion", "codigo_postal")`, y para una lista incluye el índice.
- `msg`: el mensaje legible para una persona.
- `input`: el valor que causó el error.
- `ctx`: datos adicionales del error, como el límite que se superó.
- `url`: un enlace a la documentación de ese tipo de error.

```python
from pydantic import BaseModel, ValidationError


class Item(BaseModel):
    cantidad: int


try:
    Item(cantidad="muchos")
except ValidationError as exc:
    print(exc.error_count())
    print(exc.errors())
    print(exc.json())
```

Además de `errors()`, el método `error_count()` devuelve cuántos errores hubo y `json()` los entrega ya serializados como texto JSON. Para quitar el enlace de documentación de cada error se llama a `errors(include_url=False)`.

## Errores dentro de un validador

Dentro de un `@field_validator` o un `@model_validator` se lanza `ValueError` o `AssertionError` para indicar que el dato es inválido. Pydantic captura la excepción y la convierte en un error con tipo `value_error` o `assertion_error`. El mensaje queda con el prefijo `Value error, ` seguido del texto que escribió el programador.

Una excepción distinta, como `TypeError` o `KeyError`, no se convierte en error de validación: se propaga tal cual. Es un detalle importante, porque un error de programación dentro del validador no debe quedar disfrazado de dato inválido.

## Errores personalizados con PydanticCustomError

Para controlar el tipo y el mensaje del error se usa `PydanticCustomError`, que se importa de `pydantic_core`. Recibe un código de error, una plantilla de mensaje y un diccionario de contexto con los valores que se insertan en la plantilla.

```python
from pydantic_core import PydanticCustomError

raise PydanticCustomError(
    "stock_insuficiente",
    "No hay stock suficiente: pediste {pedido} y quedan {stock}",
    {"pedido": 10, "stock": 3},
)
```

El código `stock_insuficiente` aparece en el campo `type` del error, lo que permite a un cliente reaccionar según el tipo sin interpretar el texto del mensaje.

## Cómo mostrarlos al usuario

En una API, la práctica recomendada es devolver el resultado de `errors()` con el código HTTP 422, sin exponer las excepciones internas. En una aplicación de consola conviene recorrer la lista y mostrar, para cada error, la ubicación unida con puntos y el mensaje. Nunca hay que devolver el campo `input` de errores de campos sensibles como contraseñas: para eso se usa `hide_input_in_errors=True` en `model_config`, que quita el valor de los mensajes.

## Probar los errores

En los tests se verifica el tipo de error y no el mensaje, porque el texto puede cambiar entre versiones. Con `pytest.raises(ValidationError)` se captura la excepción y se inspecciona `exc_info.value.errors()[0]["type"]`.
