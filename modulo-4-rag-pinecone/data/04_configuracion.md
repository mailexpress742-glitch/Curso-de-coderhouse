---
categoria: configuracion
etiquetas: pydantic-settings, BaseSettings, SecretStr, variables de entorno
---

# Configuración con pydantic-settings

## Un paquete aparte

En Pydantic v2 la gestión de configuración se separó en un paquete independiente llamado `pydantic-settings`. Se instala con `pip install pydantic-settings`. La clase `BaseSettings` que antes vivía en `pydantic` ahora se importa desde `pydantic_settings`. Si se intenta importar `BaseSettings` desde `pydantic` en la versión 2, Pydantic lanza un error que explica el cambio.

## BaseSettings

`BaseSettings` funciona como un `BaseModel` que además lee sus valores de variables de entorno. Cada campo se busca en el entorno por su nombre, sin distinguir mayúsculas de minúsculas. Si la variable no existe se usa el valor por defecto, y si tampoco hay valor por defecto se lanza una `ValidationError` al crear la instancia. Eso permite detectar una configuración incompleta cuando arranca la aplicación y no en medio de una petición.

```python
from pydantic_settings import BaseSettings


class Configuracion(BaseSettings):
    database_url: str
    debug: bool = False
    puerto: int = 8000


config = Configuracion()
```

Como los valores del entorno siempre son strings, Pydantic los convierte al tipo anotado: `"true"` se convierte en `True` y `"8080"` en el entero `8080`.

## SettingsConfigDict

La configuración de la clase se define en `model_config` con un `SettingsConfigDict`. Las opciones más importantes son:

- `env_file`: ruta de un archivo `.env` del que se leen las variables.
- `env_prefix`: prefijo que se agrega al nombre de cada variable. Con `env_prefix="APP_"`, el campo `puerto` se lee de `APP_PUERTO`.
- `env_nested_delimiter`: separador para configurar modelos anidados, por ejemplo `DB__HOST`.
- `extra`: qué hacer con variables sin campo asociado. Con un archivo `.env` compartido conviene `"ignore"`.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracion(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")
    puerto: int = 8000
```

## Secretos con SecretStr

Los datos sensibles, como claves de API o contraseñas, conviene declararlos con el tipo `SecretStr`. Al imprimir el valor o al incluirlo en un log se muestra como `**********`, de modo que no se filtra por accidente. Para obtener el valor real hay que llamar a `get_secret_value()` de forma explícita. También existe `SecretBytes` para datos binarios.

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Credenciales(BaseSettings):
    api_key: SecretStr
```

## Orden de prioridad de las fuentes

Cuando un mismo valor aparece en varias fuentes, gana la de mayor prioridad. El orden, de mayor a menor, es: los argumentos que se pasan al inicializar la clase, las variables de entorno, el archivo `.env`, los archivos de secretos y por último los valores por defecto. Por eso una variable exportada en el sistema tiene prioridad sobre el archivo `.env`, y es la forma de sobrescribir la configuración en producción.

## Buenas prácticas

Es recomendable crear la instancia de configuración una sola vez y reutilizarla, por ejemplo con `functools.lru_cache` en una función `get_settings()`. El archivo `.env` nunca debe subirse al repositorio: se agrega a `.gitignore` y se versiona un `.env.example` con los nombres de las variables y sin valores reales.
