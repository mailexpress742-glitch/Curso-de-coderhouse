---
categoria: tipos
etiquetas: EmailStr, HttpUrl, Decimal, Literal, Enum
---

# Tipos especiales de Pydantic v2

## Tipos de la biblioteca estándar

Pydantic entiende los tipos de la biblioteca estándar de Python sin configuración adicional. Un campo anotado con `datetime`, `date`, `time` o `timedelta` acepta objetos de ese tipo y también strings en formato ISO 8601 o números (timestamps Unix). Un campo `UUID` acepta strings con el formato de un identificador único. Un campo `Decimal` acepta números y strings, y es la opción correcta para dinero porque evita los errores de redondeo de `float`.

Para valores que solo pueden ser uno de un conjunto fijo hay dos opciones. Con `Literal["borrador", "publicado"]` el campo acepta únicamente esos strings. Con una clase que hereda de `Enum` (o de `StrEnum`) el campo acepta los valores del enumerado, y la validación convierte el string en el miembro correspondiente.

## Direcciones de correo y URLs

Para validar correos electrónicos se usa el tipo `EmailStr`. Requiere un paquete adicional, `email-validator`, que se instala con `pip install "pydantic[email]"`. Si el paquete no está instalado, Pydantic lanza un error de importación al definir el modelo.

Para URLs existen `HttpUrl`, `AnyUrl`, `AnyHttpUrl` y otros tipos específicos de esquema, como `PostgresDsn` o `RedisDsn`, que validan cadenas de conexión de bases de datos. Estos tipos no son strings: son objetos con atributos como `host`, `port` y `path`, y se convierten a texto con `str(url)`.

```python
from pydantic import BaseModel, EmailStr, HttpUrl


class Contacto(BaseModel):
    correo: EmailStr
    sitio: HttpUrl
```

## Rutas y direcciones de red

El tipo `FilePath` valida que la ruta exista y sea un archivo. `DirectoryPath` valida que exista y sea una carpeta. `IPvAnyAddress` acepta direcciones IPv4 e IPv6, y existen `IPvAnyNetwork` e `IPvAnyInterface` para redes e interfaces.

## Números restringidos

Para evitar escribir `Field(gt=0)` en cada campo hay tipos ya restringidos: `PositiveInt`, `NegativeInt`, `NonNegativeInt`, `PositiveFloat` y `NonNegativeFloat`. Para decimales con control de dígitos se usa `Field(max_digits=..., decimal_places=...)` sobre un campo `Decimal`.

## Fechas con restricciones

Los tipos `PastDate` y `FutureDate` aceptan solo fechas anteriores o posteriores al día actual. `AwareDatetime` exige que la fecha tenga zona horaria y `NaiveDatetime` exige que no la tenga. Estos tipos son útiles para evitar mezclar fechas con y sin zona horaria en un mismo sistema.

## Strict types

Los tipos `StrictInt`, `StrictStr`, `StrictBool` y `StrictFloat` desactivan la conversión automática para un solo campo, sin afectar al resto del modelo. Con `StrictInt`, el valor `"5"` se rechaza, aunque el modelo trabaje en modo lax.

## El tipo Json

El tipo `Json[list[int]]` recibe un string que contiene JSON, lo parsea y valida el contenido. Sirve para campos que llegan como texto desde una base de datos o una variable de entorno y que representan una estructura.

## Tipos personalizados

Un tipo propio se define con `Annotated` y un validador, como se explicó en la sección de validadores reutilizables. Para casos más avanzados, una clase puede implementar el método `__get_pydantic_core_schema__`, que le dice a Pydantic cómo validar y serializar ese tipo. Es la forma de integrar clases de terceros, como arreglos de una librería numérica, dentro de un modelo.
