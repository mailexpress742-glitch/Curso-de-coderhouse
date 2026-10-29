---
categoria: serializacion
etiquetas: model_dump, alias, JSON Schema, TypeAdapter
---

# Serialización y esquemas JSON en Pydantic v2

## model_dump y model_dump_json

Para convertir un modelo en datos simples se usan dos métodos. `model_dump()` devuelve un diccionario de Python, y `model_dump_json()` devuelve directamente un string JSON. Este último es más eficiente que convertir el diccionario con `json.dumps`, porque la serialización la hace el núcleo en Rust.

Ambos métodos aceptan parámetros para controlar la salida:

- `exclude` e `include`: eligen qué campos se omiten o se conservan.
- `exclude_none`: omite los campos cuyo valor es `None`.
- `exclude_unset`: omite los campos que no se asignaron explícitamente, lo que sirve para actualizaciones parciales (PATCH).
- `exclude_defaults`: omite los campos que tienen su valor por defecto.
- `by_alias`: usa el alias del campo como clave en lugar de su nombre.
- `mode`: con `"json"` el diccionario contiene solo tipos compatibles con JSON, por ejemplo fechas como strings.

## Alias

Un alias permite que el nombre del campo en los datos externos sea distinto del nombre en el código. Se define con `Field(alias="...")`. Si se quiere aceptar tanto el alias como el nombre del campo al crear instancias, se activa `populate_by_name=True` en `model_config`. Para validar con un nombre y serializar con otro existen `validation_alias` y `serialization_alias`.

```python
from pydantic import BaseModel, ConfigDict, Field


class Cliente(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    nombre_completo: str = Field(alias="fullName")
```

Para aplicar una convención a todos los campos, como pasar de snake_case a camelCase, se usa el generador `alias_generator` junto con `pydantic.alias_generators.to_camel`.

## Serializadores personalizados

El decorador `@field_serializer` cambia cómo se serializa un campo concreto. Es útil para dar formato a fechas o para ocultar parte de un valor. Para transformar el modelo completo se usa `@model_serializer`.

```python
from datetime import datetime
from pydantic import BaseModel, field_serializer


class Evento(BaseModel):
    inicio: datetime

    @field_serializer("inicio")
    def formatear(self, valor: datetime) -> str:
        return valor.strftime("%d/%m/%Y %H:%M")
```

## Esquema JSON

El método de clase `model_json_schema()` devuelve el JSON Schema del modelo. Incluye los tipos, las restricciones de `Field` y las descripciones. FastAPI usa este mecanismo para generar la documentación OpenAPI. También se usa con modelos de lenguaje: el esquema describe la estructura que el modelo debe devolver cuando se le pide una salida estructurada.

## TypeAdapter

No siempre hay una clase `BaseModel` de por medio. `TypeAdapter` permite validar y serializar cualquier tipo de Python, como `list[int]`, `dict[str, float]` o una unión de tipos. Se crea una vez y se reutiliza, porque construir el validador tiene un costo.

```python
from pydantic import TypeAdapter

adaptador = TypeAdapter(list[int])
numeros = adaptador.validate_python(["1", "2", "3"])
```

Sus métodos son `validate_python`, `validate_json`, `dump_python`, `dump_json` y `json_schema`.

## RootModel

Cuando el dato completo es un valor que no es un diccionario, como una lista, se puede usar `RootModel`. El tipo se declara con un único campo llamado `root`, y el modelo se serializa como ese valor y no como un objeto con una clave `root`.

## Modelos con herencia y uniones

Para serializar y validar uniones de modelos se recomienda una unión discriminada con `Field(discriminator="tipo")`. Pydantic lee el campo `tipo` y elige el modelo correcto sin probar todas las opciones, lo que es más rápido y da mensajes de error más claros.
