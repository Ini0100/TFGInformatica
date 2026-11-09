# Mantella AG2

Trabajo de Fin de Grado de Ingeniería Informática (Universidad San Jorge) que amplía [Mantella](https://github.com/art-from-the-machine/Mantella), el mod que permite hablar por voz con los NPC de *The Elder Scrolls V: Skyrim Special Edition* usando modelos de lenguaje. El proyecto sustituye la llamada única al LLM por una arquitectura de **agentes con AG2 (AutoGen)**, da a cada NPC una **base de conocimiento propia con RAG (ChromaDB)**, una **memoria entre conversaciones** y una **capa emocional basada en las emociones básicas de Ekman**, y todo se ejecuta **en local** con Ollama y `gemma4:e2b`.

## Funcionalidades

- **Cada NPC es un agente de AG2**: `AgentFactory` convierte el personaje de Mantella en un `AssistantAgent` con su biografía, las reglas de rol del juego, su conocimiento y su memoria. El cambio entre el motor original y el de agentes se hace con una sola opción (`use_ag2`).
- **Base de conocimiento por NPC (RAG)**: documentos fragmentados y vectorizados en ChromaDB y etiquetados con el nombre del personaje, de forma que cada NPC solo recupera lo suyo. Por ejemplo, Embry tiene cargado conocimiento sobre *One Piece* y Farengar sobre *Los Vengadores*, aunque nada de eso exista en Skyrim.
- **Recuperación en dos fases** (cliente estándar y pruebas): la pregunta del jugador se reescribe con **HyDE**, se recuperan hasta 15 fragmentos candidatos y un **Cross-Encoder en GPU** se queda con los 3 más relevantes antes de inyectarlos en el prompt.
- **Memoria entre conversaciones acotada**: al cerrar una conversación, el LLM la resume en 2-3 frases que el NPC recordará la próxima vez. El resumen sustituye al anterior en lugar de acumular el historial completo, así que el contexto no crece sin límite entre visitas.
- **Capa emocional de Ekman**: 8 estados (neutral y siete emociones básicas) con intensidad de 1 a 5, que modifican el tono del personaje sin sustituir su personalidad base.
- **Debates entre varios NPC** con `GroupChat` de AG2 y turnos en *round-robin*.
- **100 % local**: LLM con Ollama, voz con XTTS y reconocimiento de voz con Whisper, sin depender de servicios de pago.
- **Baterías de pruebas reproducibles** para cada módulo, con modo manual y modo automático que exporta las métricas a CSV.

## Stack y arquitectura

Python 3.11 · AG2 0.13 · ChromaDB · sentence-transformers · Ollama (`gemma4:e2b`) · PyTorch + CUDA · Skyrim SE + SKSE64 + Mantella 0.13.1

El repositorio corresponde a la carpeta `SKSE/Plugins` del mod. Los `.dll` de SKSE y el ejecutable de Mantella viven junto al código pero no se versionan.

```
MantellaSoftware/
├── main_custom.py              Punto de entrada: precarga ChromaDB y los modelos y levanta el servidor HTTP
├── cargar_documentos.py        Ingesta offline de un documento en la base de conocimiento de un NPC
├── src/
│   ├── ag2/
│   │   ├── agent_factory.py        Personaje de Mantella → AssistantAgent (prompt + RAG + memoria)
│   │   ├── ag2_client.py           Adapter entre la interfaz AIClient de Mantella y AG2; resume la memoria al cerrar
│   │   ├── ag2_conversation.py     Ciclo de conversación con agentes (hereda de Conversation)
│   │   └── mantella_user_proxy.py  El jugador como UserProxyAgent
│   ├── llm/
│   │   ├── my_custom_client.py     Proxy del cliente LLM con RAG: HyDE + re-ranking con Cross-Encoder
│   │   ├── gestor_memoria_ag2.py   Acceso a la base RAG y persistencia de la memoria de cada NPC
│   │   └── npc_emotions.py         NPC de prueba y capa emocional de Ekman
│   ├── conversation/           Contexto y ciclo de conversación de Mantella (adaptados)
│   ├── http/routes/            Rutas HTTP que recibe el plugin de SKSE desde el juego
│   └── game_manager.py         Elige el motor estándar o el de AG2 según use_ag2
├── test_custom_client.py       Pruebas del pipeline RAG
├── test_ag2_client.py          Pruebas de la memoria conversacional de AG2
├── test_integrated_rag_ag2.py  Pruebas de RAG + AG2 + emociones
├── test_multiagent_ag2.py      Pruebas de debates entre varios NPC
├── one_piece.txt …             Documentos de conocimiento de ejemplo
└── tests/                      Tests unitarios originales de Mantella (pytest)
```

Decisiones destacables:

- **Patrón Adapter** (`AG2Client`): implementa la misma interfaz que el cliente LLM original, de modo que el bucle del juego no se modifica y AG2 se activa o desactiva sin tocar código.
- **Aislamiento por metadatos**: todos los NPC comparten una única colección de ChromaDB (`memoria_global_npcs`), pero cada fragmento lleva el campo `npc_name` y las consultas filtran por él, para evitar que un personaje "sepa" lo que se cargó para otro.
- **Re-ranking en CUDA y fuera del event loop**: la inferencia del Cross-Encoder es síncrona, así que se ejecuta en GPU y en un hilo aparte (`asyncio.to_thread`) para no bloquear el servidor ni provocar tirones en el juego.
- **Precarga en el arranque**: los modelos pesados se cargan antes de aceptar conexiones; si se cargaran en la primera petición, el juego agotaría su timeout y cancelaría la conversación.
- **Memoria por resumen en lugar de historial**: guardar el historial completo hacía crecer el contexto en cada visita hasta saturar la VRAM y disparar la latencia en sesiones largas.

### Cómo funciona una conversación

1. El jugador habla, Whisper transcribe la frase y el plugin de SKSE la envía al servidor (`main_custom.py`).
2. Con `use_ag2` activo, `GameStateManager` delega la conversación en `AG2Conversation`.
3. `AgentFactory` crea el agente del NPC con el prompt completo de Mantella (idioma, hora, lugar, reglas de rol), su conocimiento RAG (los fragmentos etiquetados con su nombre) y el resumen de la última conversación con el jugador.
4. `AG2Client` pide la respuesta al agente con el hilo de la conversación en curso, y la respuesta pasa al TTS (XTTS) y al juego.
5. Al terminar, el LLM resume la conversación y `GestorMemoriaAG2` la guarda en `npc_conversations_db/<NPC>_memory.json`.

## Validación

Cada módulo tiene su propia batería de pruebas, ejecutable fuera del juego. Al lanzar cualquiera de ellas se puede elegir entre **[1] modo interactivo**, para escribir preguntas a mano, y **[2] batería automática**, que repite un guion fijo y guarda las métricas en `resultados_*.csv`.

| Script | Qué valida |
|---|---|
| `test_custom_client.py` | Pipeline RAG: tiempos de HyDE, recuperación y re-ranking, y puntuación del Cross-Encoder |
| `test_ag2_client.py` | Memoria de AG2: crecimiento del contexto, persistencia entre sesiones y purga por cambio de emoción |
| `test_integrated_rag_ag2.py` | RAG, AG2 y capa emocional juntos, con varios NPC y emociones |
| `test_multiagent_ag2.py` | Debates entre varios NPC: reparto de turnos, formato de respuesta y uso del contexto RAG |

Resultados principales sobre el producto final dentro del juego (5 ejecuciones por prueba):

| Aspecto evaluado | Resultado |
|---|---|
| Latencia en el juego real | 14,83 ± 0,38 s frente a 12,42 ± 1,84 s en script aislado: sobrecoste bajo y estable |
| Persistencia de memoria entre conversaciones | Funciona tras sustituir el historial completo por un resumen |
| Contradicción directa del jugador sobre un dato cargado | El NPC abandona el dato correcto en 4 de 5 ejecuciones |
| Pregunta con una premisa falsa implícita | Nunca la confirma, pero tampoco la corrige de forma consistente |

## Instalación

### Requisitos

- Windows 10/11 y una GPU NVIDIA con CUDA (desarrollado en una RTX 3050 Laptop de 4 GB).
- *Skyrim Special Edition* 1.6.1170 con [SKSE64](https://skse.silverlock.org/) 2.2.6 y el mod [Mantella](https://www.nexusmods.com/skyrimspecialedition/mods/98631) 0.13.1.
- Python 3.11.
- [Ollama](https://ollama.com) con el modelo `gemma4:e2b`.
- El servidor de voz [XTTS para Mantella](https://www.nexusmods.com/skyrimspecialedition/mods/113445).

### Instalación rápida

Desde la carpeta `MantellaSoftware`:

```powershell
py -3.11 -m venv MantellaEnv
.\MantellaEnv\Scripts\Activate
pip install -r requirements.txt
ollama pull gemma4:e2b
```

La primera ejecución descarga de Hugging Face el modelo de embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) y el Cross-Encoder de re-ranking (`mmarco-mMiniLMv2-L12-H384-v1`).

### Configuración

La configuración se edita desde la interfaz de Mantella (<http://localhost:4999/ui>, con el servidor en marcha) o directamente en `Documentos\My Games\Mantella\config.ini`:

- **LLM**: API `http://127.0.0.1:11434/v1` y modelo `gemma4:e2b`. Ollama no necesita clave; si Mantella pide `GPT_SECRET_KEY.txt`, basta con escribir cualquier valor (el fichero está ignorado por git).
- **Parámetros del LLM**: `{"max_tokens": 700}`. `gemma4:e2b` razona internamente antes de responder y ese razonamiento consume tokens del límite; con el valor por defecto de 250, las respuestas salen vacías o cortadas a mitad de frase.
- **Voz e idioma**: TTS `XTTS`, STT `Whisper` e idioma `es`.
- **Motor de agentes**: la opción `use_ag2` (*Use AG2 Agents*) viene activada por defecto.

### Ejecución

```powershell
python main_custom.py
```

Cuando el log muestre `Esperando peticiones del motor...`, abre Skyrim desde SKSE y usa el hechizo de Mantella sobre un NPC para empezar a hablar.

### Cargar conocimiento en un NPC

El nombre debe coincidir exactamente con el del personaje en el juego. El script es idempotente: al repetirlo, sustituye lo que ya tuviera cargado ese NPC.

```powershell
python cargar_documentos.py one_piece.txt Embry
python cargar_documentos.py vengadores.txt Farengar
python cargar_documentos.py star_wars.txt Nazeem
python cargar_documentos.py dragon_ball.txt Frodnar
```

### Reiniciar la memoria de los NPC

Para que los NPC olviden las conversaciones anteriores (sin perder el conocimiento RAG), borra los resúmenes guardados:

```powershell
Remove-Item npc_conversations_db\*.json
```

## Limitaciones conocidas

- `gemma4:e2b` es un modelo pequeño (5,1 B parámetros, cuantizado): tiende a dar la razón al jugador ante contradicciones y, en conversaciones de varios turnos, puede ignorar el conocimiento RAG aunque esté en el prompt.
- Con Skyrim en marcha, la GPU de 4 GB se reparte entre el juego, la voz y el modelo: cada respuesta tarda entre 10 y 15 s, y la primera tras el arranque bastante más.
- La capa emocional y los debates entre varios NPC están validados en los entornos de prueba; dentro del juego, la conversación es entre el jugador y un NPC con estado neutral.
- El modo AG2 no procesa las llamadas a herramientas (*tool calls*) que sí admite el cliente estándar de Mantella.

## Créditos y licencias

- **Código**: AGPL-3.0, la misma licencia que Mantella, del que este proyecto es una obra derivada (ver `MantellaSoftware/LICENSE`).
- **Mantella**: creado por [art-from-the-machine](https://github.com/art-from-the-machine/Mantella) y sus colaboradores.
- **Frameworks y modelos**: [AG2](https://github.com/ag2ai/ag2), [ChromaDB](https://www.trychroma.com), los modelos `paraphrase-multilingual-MiniLM-L12-v2` y `mmarco-mMiniLMv2-L12-H384-v1` de Hugging Face y `gemma4:e2b` a través de Ollama, cada uno bajo su propia licencia.
- **Datos de personajes**: Mantella usa material de los artículos *Skyrim: Characters* de la Elder Scrolls Wiki (Fandom), con licencia CC BY-SA 3.0.

*The Elder Scrolls V: Skyrim* es una marca registrada de Bethesda Softworks y ZeniMax Media. *One Piece*, *Dragon Ball*, *Star Wars* y *Los Vengadores* pertenecen a sus respectivos propietarios y aquí solo se usan como material de prueba para la base de conocimiento. Este es un proyecto académico sin ánimo de lucro, no afiliado, patrocinado ni respaldado por ninguno de ellos.
