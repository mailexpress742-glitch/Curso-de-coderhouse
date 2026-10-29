---
categoria: serializacion
etiquetas: JSON Schema, model_json_schema, strict, from_attributes
---

# Validación desde JSON y generación de esquemas

## Validar sin convertir primero

Cuando los datos llegan como un string JSON, la forma más eficiente de validarlos es `Modelo.model_validate_json(texto)`. Este método parsea y valida en un solo paso dentro del núcleo en Rust, sin crear un diccionario intermedio de Python. Además, los errores de sintaxis del JSON se informan como errores de validación, de modo que se manejan con el mismo `except ValidationError`.

Existe también `model_validate_strings`, que valida datos donde todos los valores son strings, como los que vienen de un formulario o de variables de entorno, y los convierte a los tipos del modelo.

## Validar atributos de objetos

El método `model_validate` acepta el argumento `from_attributes=True`, que le indica que lea los datos como atributos de un objeto en lugar de claves de un diccionario. Es la forma de convertir objetos de otras librerías, como filas de una base de datos, en modelos de Pydantic. La misma opción se puede dejar fija en `model_config`.

## Modo strict en una llamada

El modo strict se puede pedir en una sola llamada: `Modelo.model_validate(datos, strict=True)`. Así se usa el modo lax por defecto en el resto de la aplicación y se endurece la validación solo en los puntos donde los tipos ya deberían ser exactos. Con datos JSON, el modo strict sigue aceptando las conversiones propias de JSON, como un string con formato de fecha para un campo `datetime`.

## model_json_schema

El método de clase `model_json_schema()` devuelve un diccionario con el esquema JSON del modelo. Incluye el tipo de cada campo, cuáles son obligatorios, las restricciones de `Field` y las descripciones. Acepta el parámetro `mode`, con dos valores posibles:

- `"validation"` (el valor por defecto) describe los datos que el modelo acepta al validar.
- `"serialization"` describe los datos que el modelo produce al serializar.

La diferencia importa cuando hay campos calculados, alias o tipos que cambian al serializarse. Con `mode="serialization"` el esquema refleja la forma real de la respuesta que se envía al cliente. También se puede pasar `by_alias=False` para usar los nombres de los campos y no sus alias.

## Personalizar el esquema

Para agregar información que Pydantic no genera, se usa `json_schema_extra`. En `Field` o en `model_config` acepta un diccionario con claves adicionales, como ejemplos, o una función que modifica el esquema ya generado. Los `examples` de `Field` se incluyen en el esquema y aparecen en las herramientas de documentación.

```python
from pydantic import BaseModel, ConfigDict


class Receta(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"nombre": "Tortilla", "minutos": 20}]}
    )
    nombre: str
    minutos: int
```

## Esquemas para modelos de lenguaje

El esquema JSON de un modelo de Pydantic es la forma estándar de decirle a un modelo de lenguaje qué estructura debe devolver. Las librerías de salida estructurada envían `model_json_schema()` al proveedor y después validan la respuesta con `model_validate_json`. Si el modelo de lenguaje devuelve un JSON inválido o incompleto, la `ValidationError` indica qué campos faltan, y se puede reintentar la llamada.

## Referencias en el esquema

Cuando un modelo contiene a otros modelos, el esquema los define una sola vez en la sección `$defs` y los referencia con `$ref`. El parámetro `ref_template` permite cambiar el formato de esas referencias, algo que necesitan algunos consumidores, como ciertas versiones de OpenAPI.
