---
categoria: fundamentos
etiquetas: BaseModel, tipos, errores
---

# Modelos básicos en Pydantic v2

## Definir un modelo

En Pydantic un modelo es una clase que hereda de `BaseModel`. Los campos se declaran con anotaciones de tipo, y Pydantic usa esas anotaciones para validar y convertir los datos cuando se crea una instancia.

```python
from pydantic import BaseModel


class Usuario(BaseModel):
    id: int
    nombre: str
    email: str | None = None
```

Un campo sin valor por defecto es obligatorio. Un campo con valor por defecto, como `email`, es opcional. Si se crea `Usuario(id="7", nombre="Ana")`, Pydantic convierte el string `"7"` al entero `7`, porque por defecto trabaja en modo lax, que acepta conversiones razonables.

## Crear instancias desde datos externos

Para validar datos que vienen de afuera (un diccionario, un JSON de una API) se usan los métodos de clase `model_validate` y `model_validate_json`. El primero recibe un diccionario de Python. El segundo recibe directamente un string JSON y lo valida en un solo paso, lo que es más rápido que parsear el JSON primero y validarlo después.

```python
usuario = Usuario.model_validate({"id": 1, "nombre": "Ana"})
usuario = Usuario.model_validate_json('{"id": 1, "nombre": "Ana"}')
```

## Errores de validación

Cuando los datos no cumplen el esquema, Pydantic lanza una `ValidationError`. Esta excepción agrupa todos los errores encontrados, no solo el primero. El método `errors()` devuelve una lista con la ubicación (`loc`), el mensaje (`msg`) y el tipo (`type`) de cada error, lo que permite devolver respuestas claras a un cliente de una API.

```python
from pydantic import ValidationError

try:
    Usuario(id="abc", nombre=123)
except ValidationError as exc:
    print(exc.errors())
```

## Configuración del modelo

El comportamiento de un modelo se configura con el atributo `model_config`, al que se le asigna un `ConfigDict`. Las opciones más usadas son:

- `extra`: define qué hacer con campos que no están declarados. Con `"forbid"` se rechazan, con `"ignore"` se descartan (es el comportamiento por defecto) y con `"allow"` se conservan.
- `frozen`: con `True` el modelo es inmutable y sus instancias se pueden usar como claves de un diccionario.
- `strict`: con `True` se desactivan las conversiones automáticas, de modo que un `"7"` ya no se acepta como entero.
- `str_strip_whitespace`: con `True` se eliminan los espacios al principio y al final de los strings.

```python
from pydantic import BaseModel, ConfigDict


class Producto(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sku: str
    precio: float
```

## Modo lax y modo strict

El modo lax es el predeterminado: acepta que un string numérico se convierta en entero o que un entero se convierta en float. El modo strict exige el tipo exacto. Se puede activar para todo el modelo con `ConfigDict(strict=True)` o para un solo campo con `Field(strict=True)`. En APIs que reciben JSON conviene el modo lax, y en procesos internos donde los tipos ya están garantizados conviene el modo strict, porque detecta errores de programación.

## Modelos anidados

Un campo puede ser otro modelo. Pydantic valida de forma recursiva, y los errores indican la ruta completa hasta el campo que falló, por ejemplo `direccion.codigo_postal`. También se pueden usar listas de modelos, como `list[Producto]`, y cada elemento se valida por separado.
