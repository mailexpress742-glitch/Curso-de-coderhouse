---
categoria: modelado
etiquetas: Generic, herencia, model_rebuild, referencias
---

# Herencia, genéricos y referencias en Pydantic v2

## Herencia de modelos

Un modelo puede heredar de otro modelo. La subclase recibe todos los campos de la clase base y puede agregar campos nuevos o redefinir los existentes. Es una forma de compartir campos comunes, como un identificador y fechas de creación, entre varios modelos de una API.

```python
from datetime import datetime
from pydantic import BaseModel


class Base(BaseModel):
    id: int
    creado: datetime


class Pedido(Base):
    total: float
```

La configuración de `model_config` también se hereda. Si la subclase define su propio `model_config`, se combina con el de la clase base: las claves nuevas se agregan y las repetidas se reemplazan.

## Modelos genéricos

Un modelo genérico recibe un tipo como parámetro. Se define heredando de `BaseModel` y de `Generic` con una variable de tipo (`TypeVar`). Es el patrón habitual para envolver cualquier dato en una respuesta estándar de una API, por ejemplo con campos `datos` y `mensaje`.

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class Respuesta(BaseModel, Generic[T]):
    datos: T
    mensaje: str = "ok"


Respuesta[int](datos="3")  # datos se convierte a 3
Respuesta[list[str]](datos=["a"])
```

Cuando se parametriza el modelo, como en `Respuesta[int]`, Pydantic crea una clase nueva que valida el campo `datos` como entero. Si se usa `Respuesta` sin parámetro, el campo se valida como `Any`, es decir, sin restricción de tipo.

## Referencias hacia adelante

Cuando un modelo se refiere a una clase que todavía no está definida, la anotación se escribe como string. Pydantic resuelve la referencia más tarde. Si no puede resolverla en el momento de crear la clase, hay que llamar a `Modelo.model_rebuild()` después de definir todas las clases involucradas.

```python
class Carpeta(BaseModel):
    nombre: str
    subcarpetas: list["Carpeta"] = []
```

Los modelos recursivos, como un árbol de carpetas o de comentarios, usan esta técnica. Pydantic valida la estructura completa y detecta referencias circulares en los datos, lanzando un error en lugar de entrar en un bucle infinito.

## Campos en común con composición

La herencia no siempre es la mejor opción. Si dos modelos solo comparten uno o dos campos, conviene la composición: un modelo contiene al otro como campo. La herencia es adecuada cuando la subclase es realmente una especialización de la clase base.

## Un solo modelo con varias variantes

Una necesidad frecuente es tener un modelo para crear, otro para actualizar y otro para devolver al cliente. En lugar de repetir campos, se define una clase base con los campos compartidos y subclases que agregan o cambian lo necesario. Para el modelo de actualización parcial, donde todos los campos son opcionales, se declara cada campo con un valor por defecto `None` y se usa `model_dump(exclude_unset=True)` para obtener solo lo que el cliente envió.

## Atributos privados

Los atributos cuyo nombre empieza con guion bajo no son campos: Pydantic no los valida ni los serializa. Se declaran con `PrivateAttr` y se inicializan en el método `model_post_init`, que se ejecuta después de que el modelo se creó y validó. Sirven para guardar estado interno, como una caché de un cálculo costoso.
