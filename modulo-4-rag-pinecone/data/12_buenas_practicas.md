---
categoria: buenas-practicas
etiquetas: model_construct, validate_assignment, model_copy, PrivateAttr
---

# Buenas prácticas y trampas frecuentes

## Crear instancias sin validar

El método de clase `model_construct()` crea una instancia sin ejecutar ninguna validación ni conversión. Los valores se asignan tal cual. Es más rápido, pero peligroso: si los datos no son válidos, el modelo queda con valores incorrectos y nadie se entera. Solo se justifica cuando los datos ya fueron validados antes, por ejemplo cuando vienen de una fuente propia de confianza.

## Copiar modelos

`model_copy()` crea una copia de una instancia. Con `update={"campo": valor}` la copia sale con valores modificados, y con `deep=True` se copian también los objetos anidados. Hay que recordar que los valores de `update` no se validan: si se pasa un valor del tipo equivocado, la copia lo acepta sin protestar.

## validate_assignment

Por defecto, Pydantic valida los datos solo al crear la instancia. Si después se hace `usuario.edad = "abc"`, el modelo acepta la asignación sin validar. Con `validate_assignment=True` en `model_config`, cada asignación vuelve a validarse y lanza una `ValidationError` si el valor es inválido. La alternativa es `frozen=True`, que directamente impide modificar el modelo.

## Campos que no se deben mostrar

Para que un campo no aparezca en el `repr` del modelo se declara con `Field(repr=False)`. Para que no se incluya al serializar se usa `Field(exclude=True)`. Estas dos opciones, junto con `SecretStr`, evitan que contraseñas o claves aparezcan en logs y en respuestas.

## Campos extra

Con `extra="allow"` el modelo conserva los campos no declarados. Se pueden leer con el atributo `model_extra`, que es un diccionario con esos valores. Es útil al procesar JSON de terceros cuyo esquema cambia, aunque se pierde la garantía de que los datos tengan una forma conocida. Con `extra="forbid"` ocurre lo contrario: cualquier campo desconocido produce un error, lo que ayuda a detectar errores de tipeo en una configuración.

## Nombres reservados

Pydantic reserva el prefijo `model_` para sus propios métodos, como `model_dump` y `model_validate`. Si un campo propio empieza con ese prefijo, por ejemplo `model_name`, Pydantic emite una advertencia porque podría chocar con un método actual o futuro. Se soluciona renombrando el campo o configurando `protected_namespaces=()` en `model_config`. Es un caso frecuente en aplicaciones de IA, donde un campo `model_name` indica qué modelo de lenguaje se usa.

## Mutabilidad y valores por defecto

Un valor por defecto mutable, como una lista vacía, es seguro en Pydantic: cada instancia recibe su propia copia. Esto es distinto de lo que ocurre en una función normal de Python, donde un argumento por defecto mutable se comparte entre llamadas. Para valores por defecto calculados se usa `Field(default_factory=...)`, que ejecuta la función cada vez que se crea una instancia, por ejemplo para generar un identificador único o la fecha actual.

## Atributos privados

Los atributos con guion bajo inicial se declaran con `PrivateAttr`. No son campos: no se validan, no se serializan y no aparecen en el esquema. El método `model_post_init` se ejecuta al terminar la inicialización y es el lugar para preparar ese estado interno.

## Rendimiento

Para sacar partido del núcleo en Rust conviene usar `model_validate_json` en lugar de parsear el JSON a mano, crear los `TypeAdapter` una sola vez, y preferir tipos concretos antes que `Any`. Si una validación se repite millones de veces, medir antes de optimizar: casi siempre el cuello de botella está en la entrada y salida de datos, no en la validación.
