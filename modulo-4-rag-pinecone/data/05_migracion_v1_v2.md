---
categoria: migracion
etiquetas: pydantic-core, bump-pydantic, Rust, rendimiento
---

# Migración de Pydantic v1 a v2 y rendimiento

## Por qué existe la versión 2

Pydantic v2 reescribió la lógica de validación en un componente llamado `pydantic-core`, que está implementado en Rust. El resultado es una validación y una serialización mucho más rápidas que en la versión 1, que estaba escrita en Python. En muchos casos la mejora es de entre cinco y cincuenta veces, según el modelo. La API también se ordenó: se renombraron métodos para que todos los que pertenecen al modelo empiecen con `model_`, lo que evita choques con los nombres de los campos del usuario.

## Tabla de equivalencias

Los cambios de nombres más comunes al migrar son:

- `parse_obj()` pasó a ser `model_validate()`.
- `parse_raw()` pasó a ser `model_validate_json()`.
- `dict()` pasó a ser `model_dump()`.
- `json()` pasó a ser `model_dump_json()`.
- `schema()` pasó a ser `model_json_schema()`.
- `copy()` pasó a ser `model_copy()`.
- `@validator` pasó a ser `@field_validator`.
- `@root_validator` pasó a ser `@model_validator`.
- La clase interna `Config` se reemplazó por el atributo `model_config`.
- `Field(regex=...)` pasó a ser `Field(pattern=...)`.
- Los campos `Optional[X]` ya no tienen un valor por defecto implícito de `None`: ahora un campo opcional sin valor por defecto es obligatorio, aunque acepte `None`.

Los nombres viejos siguen funcionando en la versión 2, pero emiten un aviso de deprecación y se eliminarán en una versión futura.

## La herramienta bump-pydantic

Para migrar un proyecto grande existe `bump-pydantic`, una herramienta de línea de comandos que reescribe automáticamente el código fuente aplicando los cambios más habituales, como los renombres de métodos y de decoradores. Se instala con `pip install bump-pydantic` y se ejecuta sobre la carpeta del proyecto. No resuelve todos los casos, así que después hay que revisar el resultado y correr la suite de tests.

## Cambios de comportamiento

Algunas diferencias no son de nombres sino de comportamiento:

- La conversión de tipos es más estricta en ciertos casos. Por ejemplo, un `float` con parte decimal ya no se convierte en `int`.
- Los validadores de campo ahora reciben un parámetro `info` en lugar de `values`, y el acceso a otros campos se hace con `info.data`.
- Los modelos ya no son comparables con diccionarios.
- Los tipos genéricos se manejan con una sintaxis más simple, y las clases genéricas se definen heredando de `BaseModel` y `Generic`.

## Cómo aprovechar el rendimiento

Para obtener el mejor rendimiento conviene seguir algunas prácticas. Usar `model_validate_json` en lugar de parsear el JSON y después validarlo evita crear objetos intermedios. Crear una sola vez los objetos `TypeAdapter` y reutilizarlos evita reconstruir el validador en cada llamada. Preferir tipos concretos como `list[int]` en lugar de `Any` permite que el núcleo en Rust haga todo el trabajo sin volver a Python. Y los validadores escritos en Python son más lentos que las restricciones declarativas de `Field`, así que conviene usarlos solo cuando hacen falta.

## Compatibilidad durante la transición

Si un proyecto no puede migrar de una vez, Pydantic 2 incluye el espacio de nombres `pydantic.v1`, que contiene la API completa de la versión 1. Permite mantener modelos antiguos mientras se migra de a poco, aunque un modelo de v1 no puede mezclarse dentro de un modelo de v2.
