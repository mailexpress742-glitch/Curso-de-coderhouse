---
categoria: integraciones
etiquetas: FastAPI, response_model, 422, from_attributes
---

# Integración de Pydantic con FastAPI

## FastAPI se apoya en Pydantic

FastAPI usa Pydantic para validar los datos de entrada y para describir los datos de salida. Cuando una función de una ruta declara un parámetro anotado con un modelo, FastAPI lee el cuerpo JSON de la petición, lo valida con ese modelo y entrega a la función una instancia ya validada. Si la función necesita el dato, lo recibe tipado y sin conversiones manuales.

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


class ProductoCrear(BaseModel):
    nombre: str = Field(min_length=3)
    precio: float = Field(gt=0)


@app.post("/productos")
def crear(producto: ProductoCrear):
    return producto
```

## El error 422

Cuando los datos de la petición no cumplen el modelo, FastAPI no ejecuta la función. Responde con el código de estado 422 (Unprocessable Entity) y un cuerpo JSON con la clave `detail`, que contiene la lista de errores de la `ValidationError`: la ubicación de cada error (`loc`), el mensaje (`msg`) y el tipo (`type`). El cliente de la API recibe así un diagnóstico preciso sin que el programador escriba código de manejo de errores.

## response_model

El parámetro `response_model` de un decorador de ruta indica qué modelo describe la respuesta. FastAPI filtra el valor que devuelve la función para que contenga solo los campos de ese modelo y lo valida antes de enviarlo. Es la forma de no filtrar datos sensibles: se define un modelo de entrada que incluye la contraseña y un modelo de salida que no la incluye.

```python
class UsuarioCrear(BaseModel):
    email: str
    password: str


class UsuarioLeer(BaseModel):
    id: int
    email: str


@app.post("/usuarios", response_model=UsuarioLeer)
def crear_usuario(datos: UsuarioCrear):
    return {"id": 1, "email": datos.email, "password": datos.password}
```

La respuesta de este ejemplo no incluye `password`, porque `UsuarioLeer` no tiene ese campo. Las opciones `response_model_exclude_unset` y `response_model_exclude_none` controlan si se omiten los campos sin asignar o con valor `None`.

## Objetos de una base de datos

Los objetos que devuelve un ORM, como SQLAlchemy, no son diccionarios sino instancias con atributos. Para que un modelo de Pydantic pueda leerlos se activa `from_attributes=True` en `model_config`. En la versión 1 de Pydantic esta opción se llamaba `orm_mode`. Con ella, `Modelo.model_validate(objeto_orm)` lee los atributos del objeto como si fueran claves.

## Parámetros de consulta y de ruta

FastAPI también valida los parámetros de ruta y de consulta (query) usando las anotaciones de tipo de la función. Para restricciones se usan `Query` y `Path` con los mismos argumentos que `Field`, como `ge`, `le` y `min_length`. Un modelo de Pydantic puede agrupar varios parámetros de consulta si se declara con `Annotated[Modelo, Query()]`.

## Documentación automática

Con los modelos declarados, FastAPI genera el esquema OpenAPI a partir de `model_json_schema()` y publica una documentación interactiva en las rutas `/docs` y `/redoc`. Las descripciones y ejemplos definidos con `Field(description=..., examples=[...])` aparecen en esa documentación.

## Configuración como dependencia

La configuración de la aplicación suele ser una clase `BaseSettings`. Se crea una función `get_settings()` decorada con `lru_cache` y se inyecta en las rutas con `Depends(get_settings)`. Así, el archivo `.env` se lee una sola vez y las pruebas pueden reemplazar la configuración sobrescribiendo la dependencia.
