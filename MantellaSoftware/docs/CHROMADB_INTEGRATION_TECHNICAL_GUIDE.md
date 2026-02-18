# Guía Técnica: Integración de ChromaDB en Mantella

Esta guía detalla los cambios de código necesarios para reemplazar o extender el sistema de memoria actual (basado en archivos de texto) con una base de datos vectorial ChromaDB.

## Objetivo
Implementar una memoria a largo plazo semántica que permita a los NPCs recordar detalles específicos relevantes para la conversación actual, en lugar de cargar un resumen genérico.

## Dependencias
Se requiere instalar la librería `chromadb`:
```bash
pip install chromadb
```

## Arquitectura Propuesta

El sistema actual utiliza la clase `Summaries` (en `src/remember/summaries.py`) que hereda de `Remembering`. La mejor estrategia es crear una nueva implementación de `Remembering` llamada `ChromaMemory`.

### 1. Nueva Clase `ChromaMemory`

Crear un nuevo archivo `src/remember/chroma_memory.py`.

```python
from src.remember.remembering import Remembering
import chromadb
from chromadb.utils import embedding_functions

class ChromaMemory(Remembering):
    def __init__(self, path="data/chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        # Usar un modelo de embedding ligero por defecto, o configurar uno específico
        self.collection = self.client.get_or_create_collection(name="npc_memories")

    def get_prompt_text(self, npcs_in_conversation: Characters, world_id: str) -> str:
        # 1. Obtener el contexto actual (ej. última frase del jugador o resumen breve)
        # 2. Consultar ChromaDB para cada NPC en la conversación
        results = self.collection.query(
            query_texts=[current_context],
            n_results=5,
            where={"npc_name": {"$in": [npc.name for npc in npcs_in_conversation.get_all_characters()]}}
        )
        # 3. Formatear los resultados en un string para el prompt
        return self._format_memories(results)

    def save_conversation_state(self, messages: message_thread, npcs_in_conversation: Characters, world_id: str, is_reload=False):
        # 1. Extraer los nuevos mensajes de la conversación
        # 2. Añadirlos a la colección con metadatos
        for msg in new_messages:
            self.collection.add(
                documents=[msg.content],
                metadatas=[{"npc_name": npc.name, "world_id": world_id, "role": msg.role}],
                ids=[f"{npc.name}_{timestamp}"]
            )
```

### 2. Integración en el Flujo Principal

Es necesario modificar donde se instancia la clase de memoria (probablemente en `main.py` o `src/game_manager.py` o donde se inyecte `Summaries`).

**Archivo a modificar:** `src/game_manager.py` (o similar, buscar referencias a `Summaries`).

```python
# Pseudocódigo del cambio
if config.use_chromadb:
    self.memory_system = ChromaMemory()
else:
    self.memory_system = Summaries(...)
```

### 3. Configuración

Añadir una nueva opción en `src/config/definitions/other_definitions.py` (o `memory_definitions.py` si existiera) para activar/desactivar ChromaDB.

## Desafíos y Consideraciones

1.  **Contexto de Consulta (`query_texts`)**: En `get_prompt_text`, ¿qué usamos para buscar en la memoria?
    *   *Opción A:* La última entrada del usuario.
    *   *Opción B:* Un resumen de los últimos 3 mensajes.
    *   *Recomendación:* Empezar con la última entrada del usuario.

2.  **Persistencia**: Asegurarse de que la ruta de la base de datos (`path="data/chroma_db"`) sea persistente y relativa a la carpeta del mod.

3.  **Embeddings**: ChromaDB usa `all-MiniLM-L6-v2` por defecto. Es ligero y funciona bien en CPU, ideal para este mod.

4.  **Migración**: Las memorias existentes en `.txt` no se pasarán automáticamente a ChromaDB a menos que se escriba un script de migración.

## Pasos de Implementación

1.  Instalar `chromadb`.
2.  Crear `src/remember/chroma_memory.py` con la lógica básica.
3.  Modificar la instanciación de la memoria para usar `ChromaMemory` cuando se active una flag (puede ser hardcoded para pruebas iniciales).
4.  Probar que las conversaciones se guardan en la DB y se recuperan contextos relevantes.

---

# Actualización: Implementación Final

Esta sección detalla cómo se realizó finalmente la implementación, destacando las diferencias y soluciones técnicas adoptadas respecto a la propuesta original.

## Arquitectura Implementada

### 1. Clase `ChromaMemory` (`src/remember/chroma_memory.py`)

Se implementó la clase heredando de `Remembering`, pero con cambios clave en la lógica de filtrado y almacenamiento.

#### Guardado (`save_conversation_state`)
- **ID Determinista**: En lugar de timestamps, se usa `md5(contenido_mensaje + world_id)` para prevenir duplicados exactos.
- **Metadatos de NPCs**: No se guarda un solo nombre. Se guarda una **lista de strings** (`npc_names`) con *todos* los NPCs presentes en la conversación. Esto permite que una memoria pertenezca a varios personajes a la vez.

#### Recuperación (`get_prompt_text`)
- **Filtrado Avanzado**: La consulta original con `$in` no era suficiente para listas. Se implementó una lógica usando `$contains` con operadores booleanos:
    ```python
    where_clause = {
        "$and": [
            {"world_id": world_id},
            {"$or": [{"npc_names": {"$contains": name}} for name in npc_names]}
        ]
    }
    ```
    Esto asegura que se recuperen memorias donde aparezca *al menos uno* de los NPCs actuales.

### 2. Integración y Configuración

- **`src/game_manager.py`**: Se implementó la instanciación condicional tal cual se propuso.
- **`config.ini`**: Se añadió `use_chromadb = True` en la sección `[Other]`.

## Desafíos Resueltos

1.  **Sintaxis de Consulta**: La mayor dificultad fue filtrar correctamente por múltiples NPCs. La solución fue usar metadatos de tipo lista y operadores lógicos explícitos (`$and`, `$or`, `$contains`).
2.  **Versión de ChromaDB**: Se confirmó que la versión instalada soporta estas operaciones avanzadas de filtrado.

## Estado
Funcionalidad completa, verificada con tests automatizados y lista para producción.
