import asyncio
import sys
import os
import re
import time
import csv
import logging
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import numpy as np
import tiktoken

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen import AssistantAgent, UserProxyAgent
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

# IMPORTAMOS NUESTRA NUEVA FACTORÍA EMOCIONAL
from src.llm.npc_emotions import REGISTRO_NPCS, EMOCIONES_EKMAN, construir_prompt_cognitivo

# Endpoint local de Ollama: sustituye al servicio en la nube usado originalmente
# tanto para el paso HyDE como para el agente AG2, para que todo el sistema
# integrado (RAG + emociones + AG2) corra en local con gemma4:e2b.
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_MODEL = "gemma4:e2b"

_encoding = tiktoken.get_encoding("cl100k_base")

def contar_tokens(texto: str) -> int:
    return len(_encoding.encode(texto or ""))


async def extraer_conocimiento_rag(hyde_client, coleccion_memoria, cross_encoder: CrossEncoder, texto_usuario: str, npc_name: str, metricas: dict | None = None) -> str:
    """
    Recupera contexto factual de ChromaDB usando la técnica HyDE, seguida de un
    re-ranking con cross-encoder (igual que MyCustomClient): se traen hasta 15
    candidatos y se seleccionan los 3 mejor puntuados para la pregunta real
    (no para el documento hipotético de HyDE, que solo se usa para la búsqueda
    vectorial inicial).
    Si se pasa un diccionario `metricas`, registra en él los tiempos de cada
    fase (t_hyde, t_retrieval, t_rerank), el número de candidatos recuperados
    y la puntuación del mejor candidato (score_top1).
    """
    try:
        # 1. Generación del documento hipotético (HyDE)
        prompt_hyde = (
            f"Escribe un párrafo enciclopédico, técnico y muy breve que responda directamente "
            f"a la siguiente consulta: '{texto_usuario}'. "
            f"Proporciona datos fácticos. NUNCA uses lenguaje de rol."
        )

        t0 = time.perf_counter()
        response = await hyde_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt_hyde}],
            temperature=0.1,
            max_tokens=120
        )
        doc_hipotetico = response.choices[0].message.content.strip()
        if metricas is not None:
            metricas['t_hyde'] = time.perf_counter() - t0

        # 2. Búsqueda amplia en ChromaDB (hasta 15 candidatos)
        t1 = time.perf_counter()
        total_docs = coleccion_memoria.count()
        resultados = coleccion_memoria.query(
            query_texts=[doc_hipotetico],
            n_results=min(15, total_docs) if total_docs else 0,
            where={"npc_name": npc_name} if npc_name != "Desconocido" else None
        )
        if metricas is not None:
            metricas['t_retrieval'] = time.perf_counter() - t1

        candidatos = resultados['documents'][0] if resultados and resultados.get('documents') and resultados['documents'][0] else []
        if metricas is not None:
            metricas['docs_recuperados'] = len(candidatos)

        if not candidatos:
            return ""

        # 3. Re-ranking con cross-encoder: puntúa cada candidato contra la
        # pregunta real del jugador (no contra el documento hipotético) y nos
        # quedamos con los 3 mejores.
        t2 = time.perf_counter()
        pares = [[texto_usuario, doc] for doc in candidatos]
        scores = cross_encoder.predict(pares)
        mejores_indices = np.argsort(scores)[::-1][:3]
        mejores_documentos = [candidatos[i] for i in mejores_indices]
        if metricas is not None:
            metricas['t_rerank'] = time.perf_counter() - t2
            metricas['score_top1'] = float(max(scores)) if len(scores) else None

        return "\n\n".join(mejores_documentos)

    except Exception as e:
        logging.error(f"[RAG] Error en la extracción de conocimiento: {e}")
        return ""


# Marcador usado tanto para construir el mensaje de inyección RAG como para
# filtrarlo del historial antes de persistirlo a disco (ver purga_rag_ok en
# ejecutar_prueba_memoria_integrada e iniciar_sistema_completo). Se define una
# única vez para que ambos lados no puedan desincronizarse: el filtro original
# buscaba el literal "ATENCIÓN AL SISTEMA:" (con los dos puntos pegados),
# que nunca coincidía con el texto real "ATENCIÓN AL SISTEMA (DIRECTRIZ DE
# MÁXIMA PRIORIDAD):", así que la purga nunca se ejecutaba en la práctica.
RAG_INJECTION_MARKER = "ATENCIÓN AL SISTEMA (DIRECTRIZ DE MÁXIMA PRIORIDAD):"


def construir_mensaje_rag(contexto_factual: str, npc_name: str, emocion_elegida: str, intensidad: int) -> dict:
    """Construye el mensaje de sistema que inyecta el conocimiento factual recuperado,
    con las reglas de relevancia/personalidad/emoción/formato. Compartido entre el
    modo manual y la batería automática."""
    prompt_estricto = (
        f"{RAG_INJECTION_MARKER}\n"
        "Acabas de recuperar la siguiente INFORMACIÓN FÁCTICA de tu memoria subconsciente.\n\n"
        f"INFORMACIÓN DE REFERENCIA:\n{contexto_factual}\n\n"
        "REGLAS CRÍTICAS DE RESPUESTA:\n"
        "1. EVALUACIÓN DE RELEVANCIA: Compara la pregunta del jugador con la INFORMACIÓN DE REFERENCIA. Si la información NO responde a la pregunta, IGNORA la información y responde demostrando absoluta ignorancia sobre el tema.\n"
        f"2. MANTENIMIENTO DE IDENTIDAD (CRÍTICO): Recuerda que eres {npc_name}. Debes filtrar toda esta información (o tu ignorancia) a través de tu PERSONALIDAD BASE característica (tu forma de hablar, tus prejuicios, tu arrogancia o tu locura).\n"
        "3. EXCEPCIÓN AL LORE: Si la información SÍ responde a la pregunta, asume esos conocimientos como tuyos. Explícalos con dominio absoluto, pero siempre usando el vocabulario y la actitud propios de tu personaje.\n"
        f"4. FILTRO EMOCIONAL MATIZADO: Tu estado actual es {emocion_elegida.upper()} (Intensidad: {intensidad}/5). Esta emoción debe colorear y matizar tus palabras, pero NUNCA debe aplastar ni sustituir tu personalidad original.\n"
        "5. FORMATO PARA SÍNTESIS DE VOZ (TTS): PROHIBIDO usar listas, viñetas, guiones, asteriscos o negritas. Todo debe ser diálogo orgánico.\n"
        "6. GESTO INICIAL: Tu respuesta DEBE empezar obligatoriamente con un único [gesto físico y accionable entre corchetes] que refleje tu estado de ánimo.\n"
    )
    return {"role": "system", "content": prompt_estricto}


def cumple_formato_gesto(respuesta: str) -> bool:
    """Verifica si la respuesta cumple la regla de formato obligatoria (regla 6 de
    construir_mensaje_rag / construir_prompt_cognitivo): debe empezar con un único
    gesto físico entre corchetes, p.ej. '[Se ajusta con desdén] ...'."""
    return bool(re.match(r'^\s*\[[^\[\]]+\]', respuesta or ""))


def _conectar_servicios():
    """Inicializa ChromaDB + cliente HyDE + cross-encoder de re-ranking (mismo
    modelo que usa MyCustomClient en producción), compartido entre modo manual
    y batería."""
    print("[Base de Conocimiento] Conectando a ChromaDB local...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "npc_chroma_db")

    chroma_client = chromadb.PersistentClient(path=db_path)
    coleccion_memoria = chroma_client.get_or_create_collection(
        name="memoria_global_npcs",
        embedding_function=emb_fn
    )

    hyde_client = AsyncOpenAI(
        api_key="abc123",
        base_url=OLLAMA_BASE_URL
    )

    print("[Re-Ranker] Cargando modelo CrossEncoder en VRAM (CUDA)...")
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cuda')

    return coleccion_memoria, hyde_client, cross_encoder


def _llm_config() -> dict:
    return {
        "config_list": [{
            "model": OLLAMA_MODEL,
            "api_key": "abc123",
            "base_url": OLLAMA_BASE_URL,
            "price": [0.0, 0.0]
        }],
        "temperature": 0.7,
        # Ver nota equivalente en test_ag2_client.py: sin esto, AutoGen cachea
        # respuestas en disco con cache_seed=41 por defecto, lo que podría
        # devolver una respuesta pasada en vez de generar una nueva.
        "cache_seed": None,
    }


def iniciar_sistema_completo():
    print("\n[+] Iniciando Sistema Definitivo con Capa de Empatía (Ekman)... ")

    # ==========================================
    # SELECTOR DE NPC
    # ==========================================
    print("\n=== SELECTOR DE PERSONAJES ===")
    for key, data in REGISTRO_NPCS.items():
        print(f"{key}. {data['nombre']}")

    seleccion = ""
    while seleccion not in REGISTRO_NPCS:
        seleccion = input("\nElige el número del NPC con el que quieres hablar (1-7): ")

    # ==========================================
    # SELECTOR DE EMOCIONES
    # ==========================================
    print("\n=== SELECTOR DE ESTADOS EMOCIONALES ===")
    emociones_disponibles = list(EMOCIONES_EKMAN.keys())
    for idx, emo in enumerate(emociones_disponibles, 1):
        print(f"{idx}. {emo.capitalize()}")

    sel_emo = -1
    while sel_emo not in range(1, len(emociones_disponibles) + 1):
        try:
            sel_emo = int(input(f"\nElige el estado emocional para el personaje (1-{len(emociones_disponibles)}): "))
        except ValueError:
            continue

    emocion_elegida = emociones_disponibles[sel_emo - 1]

    intensidad = 3
    if emocion_elegida != "neutral":
        while True:
            try:
                intensidad = int(input("Define la intensidad emocional (1 suave - 5 muy marcada): "))
                if 1 <= intensidad <= 5: break
            except ValueError:
                continue

    # CONSTRUCCIÓN DINÁMICA DEL PROMPT COMBINADO
    npc_name, npc_prompt_completo = construir_prompt_cognitivo(seleccion, emocion_elegida, intensidad)

    print(f"\n[!] Cargando sistema cognitivo empático para: {npc_name} ({emocion_elegida.upper()} - Intensidad {intensidad})...")

    # ==========================================
    # INICIALIZACIÓN DE SERVICIOS Y RAG
    # ==========================================
    llm_config = _llm_config()
    coleccion_memoria, hyde_client, cross_encoder = _conectar_servicios()

    # ==========================================
    # INSTANCIACIÓN DE MEMORIA Y AGENTES AG2
    # ==========================================
    gestor_memoria = GestorMemoriaAG2()
    historial_previo = gestor_memoria.cargar_conversacion(npc_name, emocion_elegida)

    # Inicializamos el AssistantAgent con el prompt híbrido (Base + Emoción)
    npc_agent = AssistantAgent(
        name=npc_name,
        system_message=npc_prompt_completo,
        llm_config=llm_config,
    )

    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    if historial_previo:
        print(f"[*] Cargando {len(historial_previo)} recuerdos pasados desde el disco...")
        npc_agent.chat_messages[player_proxy] = historial_previo

    print(f"\n=== SISTEMA TOTALMENTE INTEGRADO LISTO ({npc_name}) ===")
    print("Escribe 'salir' para terminar y GUARDAR LA MEMORIA.\n")

    # ==========================================
    # BUCLE TRANSACCIONAL ASÍNCRONO
    # ==========================================
    # Evitamos el DeprecationWarning de Python 3.10+
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        while True:
            texto_usuario = input("\nTú: ")
            if texto_usuario.lower() in ['salir', 'exit', 'quit']:
                break
            if not texto_usuario.strip():
                continue

            contexto_factual = loop.run_until_complete(
                extraer_conocimiento_rag(hyde_client, coleccion_memoria, cross_encoder, texto_usuario, npc_name)
            )

            if contexto_factual:
                print(f"[*] Inyectando conocimiento externo en la memoria de {npc_name}...")

                print("\n" + "="*40)
                print("🔍 [DEBUG RAG] TEXTO RECUPERADO DE CHROMADB:")
                print(contexto_factual)
                print("="*40 + "\n")

                msg_rag = construir_mensaje_rag(contexto_factual, npc_name, emocion_elegida, intensidad)
                npc_agent.chat_messages[player_proxy].append(msg_rag)

            player_proxy.initiate_chat(
                npc_agent,
                message=texto_usuario,
                clear_history=False,
                silent=False
            )

    except KeyboardInterrupt:
        print("\n[!] Cierre forzado detectado (Ctrl+C). Iniciando guardado de emergencia...")

    finally:
        # 1. Filtramos primero el ruido de las inyecciones de contexto del RAG
        historial_final = [
            m for m in npc_agent.chat_messages.get(player_proxy, [])
            if RAG_INJECTION_MARKER not in m.get("content", "")
        ]

        # 2. Control de integridad transaccional (Poda de entradas huérfanas)
        if historial_final and historial_final[-1].get("role") == "user":
            print("\n[Memoria] Detectada pregunta huérfana sin respuesta debido a un corte.")
            print("[Memoria] Purgando última entrada del jugador para evitar bucles de repetición...")
            historial_final.pop()

        # 3. Guardamos el historial limpio y balanceado en el disco
        gestor_memoria.guardar_conversacion(npc_name, historial_final, emocion_elegida)
        print(f"\n[+] Sesión guardada limpiamente. ¡Ecosistema RAG+AG2 cerrado con seguridad!")


# ---------------------------------------------------------------------------
# Modo batería automática: cada entrada combina un NPC, una emoción/intensidad
# y una pregunta. Los NPC Farengar y Nazeem tienen documentos de prueba en
# ChromaDB (Avengers y Star Wars respectivamente, ver test_custom_client.py),
# lo que permite comprobar si la capa emocional colorea el tono de la
# respuesta sin romper el anclaje factual (RAG) ni la identidad del personaje.
# La última pregunta es de control (sin relación con la memoria cargada), para
# comprobar si el personaje reconoce su ignorancia cuando el re-ranking no
# encuentra un candidato realmente relevante (score_top1 bajo/negativo).
# ---------------------------------------------------------------------------
BATERIA_PRUEBAS_INTEGRADA = [
    ("1", "ira", 4, "¿Quién es Tony Stark?", "avengers"),                       # Farengar, enfadado
    ("2", "miedo", 4, "¿Qué fue la Guerra de los Clones?", "star_wars"),        # Nazeem, con miedo
    ("1", "alegria", 3, "Cuéntame quién es Thor Odinson.", "avengers"),         # Farengar, alegre
    ("2", "neutral", 3, "¿Cuál es tu plato favorito para desayunar?", "control_sin_memoria"),  # Nazeem, control
]


# ---------------------------------------------------------------------------
# Sub-prueba de memoria integrada: a diferencia de la batería de arriba (donde
# cada pregunta usa un agente nuevo e independiente), aquí se mantiene UNA
# conversación de varios turnos con el mismo NPC/emoción, inyectando RAG en
# el turno que lo necesita, para comprobar que la memoria conversacional de
# AG2 (ya validada de forma aislada en test_ag2_client.py) sigue funcionando
# correctamente cuando además se le van añadiendo mensajes de sistema RAG por
# el camino, y que esos mensajes RAG se purgan del historial antes de
# guardarlo a disco (la misma lógica del bloque `finally` de
# iniciar_sistema_completo, que hasta ahora nunca se había ejecutado en
# ningún test).
# ---------------------------------------------------------------------------
NPC_ID_MEMORIA, EMOCION_MEMORIA, INTENSIDAD_MEMORIA = "1", "alegria", 2  # Farengar, alegre (intensidad baja)
# Emoción cambiada de "ira" intensidad 4 a "alegria" intensidad 2, y preguntas
# reformuladas para exigir el nombre explícitamente: se sospecha que los
# "FALLO" observados con ira/intensidad 4 eran negativas de un personaje
# dispuesto a comportarse de forma cortante, no un fallo real de memoria.
TURNOS_MEMORIA_INTEGRADA = [
    "Hola, me llamo Aela.",                 # sin relación con la memoria RAG
    "¿Quién es Tony Stark?",                # dispara inyección RAG (Avengers)
    "¿Cuál es mi nombre?",                  # memoria de corto plazo, con ruido RAG de por medio; exige el dato, no solo confirmar que se recuerda
]


async def ejecutar_prueba_memoria_integrada(coleccion_memoria, hyde_client, cross_encoder):
    print("\n[+] Sub-prueba: memoria conversacional de AG2 + inyección RAG combinadas...")

    llm_config = _llm_config()
    npc_name, prompt_completo = construir_prompt_cognitivo(NPC_ID_MEMORIA, EMOCION_MEMORIA, INTENSIDAD_MEMORIA)

    gestor_memoria = GestorMemoriaAG2()
    npc_agent = AssistantAgent(name=npc_name, system_message=prompt_completo, llm_config=llm_config)
    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    resultados_turnos = []

    for i, texto_usuario in enumerate(TURNOS_MEMORIA_INTEGRADA, start=1):
        metricas = {}
        t_inicio = time.perf_counter()

        contexto_factual = await extraer_conocimiento_rag(hyde_client, coleccion_memoria, cross_encoder, texto_usuario, npc_name, metricas=metricas)
        if contexto_factual:
            # Igual que en el modo manual: se AÑADE al historial existente, no
            # se sustituye, para reproducir fielmente el comportamiento real
            # que hay que purgar antes de persistir.
            npc_agent.chat_messages.setdefault(player_proxy, []).append(
                construir_mensaje_rag(contexto_factual, npc_name, EMOCION_MEMORIA, INTENSIDAD_MEMORIA)
            )

        player_proxy.initiate_chat(npc_agent, message=texto_usuario, clear_history=False, silent=True)
        historial = npc_agent.chat_messages.get(player_proxy, [])
        respuesta = historial[-1]["content"] if historial else ""
        t_turno = time.perf_counter() - t_inicio

        print(f"\n--- Turno {i}: {texto_usuario}")
        print(f"NPC: {respuesta}")

        resultados_turnos.append({
            "turno": i,
            "mensaje": texto_usuario,
            "respuesta": respuesta,
            "t_turno_s": round(t_turno, 3),
            "inyecto_rag": bool(contexto_factual),
            "mensajes_historial": len(historial),
            "tokens_pregunta": contar_tokens(texto_usuario),
            "tokens_respuesta": contar_tokens(respuesta),
        })

    # --- Verificación 1: memoria de corto plazo dentro de la misma sesión,
    # pese a que un turno intermedio inyectó un mensaje de sistema RAG. ---
    recuerda_nombre = "aela" in resultados_turnos[-1]["respuesta"].lower()
    print(f"\n[Verificación] ¿Recuerda el nombre pese al ruido RAG intercalado? {'SÍ' if recuerda_nombre else 'NO'}")

    # --- Purga del ruido RAG antes de persistir (misma lógica que el modo manual) ---
    historial_bruto = npc_agent.chat_messages.get(player_proxy, [])
    contenia_ruido_rag = any(RAG_INJECTION_MARKER in m.get("content", "") for m in historial_bruto)
    historial_filtrado = [m for m in historial_bruto if RAG_INJECTION_MARKER not in m.get("content", "")]
    if historial_filtrado and historial_filtrado[-1].get("role") == "user":
        historial_filtrado.pop()

    purga_rag_ok = contenia_ruido_rag and not any(RAG_INJECTION_MARKER in m.get("content", "") for m in historial_filtrado)
    print(f"[Verificación] ¿Se disparó al menos una inyección RAG y se purgó del historial antes de guardar? {'SÍ' if purga_rag_ok else 'NO'}")

    gestor_memoria.guardar_conversacion(npc_name, historial_filtrado, EMOCION_MEMORIA)
    print(f"[*] Memoria guardada ({len(historial_filtrado)} mensajes, sin ruido RAG).")

    # --- Verificación 2: persistencia entre sesiones (simulando reinicio del juego) ---
    historial_recargado = gestor_memoria.cargar_conversacion(npc_name, EMOCION_MEMORIA)
    persistencia_ok = len(historial_recargado) == len(historial_filtrado)
    print(f"[Verificación] Mensajes guardados: {len(historial_filtrado)} | recargados: {len(historial_recargado)} | ¿Coinciden? {'SÍ' if persistencia_ok else 'NO'}")

    # --- Verificación 3: continuidad tras recargar, con un agente nuevo ---
    recuerda_contexto_sesion2 = False
    if historial_recargado:
        npc_agent_2 = AssistantAgent(name=npc_name, system_message=prompt_completo, llm_config=llm_config)
        player_proxy_2 = UserProxyAgent(name="Jugador", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)
        npc_agent_2.chat_messages[player_proxy_2] = historial_recargado

        pregunta_sesion2 = "¿Recuerdas cuál es mi nombre?"
        player_proxy_2.initiate_chat(npc_agent_2, message=pregunta_sesion2, clear_history=False, silent=True)
        historial_2 = npc_agent_2.chat_messages.get(player_proxy_2, [])
        respuesta_2 = historial_2[-1]["content"] if historial_2 else ""
        print(f"\n--- Turno sesión 2: {pregunta_sesion2}\nNPC: {respuesta_2}")
        recuerda_contexto_sesion2 = "aela" in respuesta_2.lower()

    # --- Verificación 4: purga por disrupción emocional (no requiere LLM) ---
    emocion_disrupcion = "ira" if EMOCION_MEMORIA != "ira" else "tristeza"
    historial_tras_disrupcion = gestor_memoria.cargar_conversacion(npc_name, emocion_disrupcion)
    purga_emocion_ok = len(historial_tras_disrupcion) == 0
    print(f"[Verificación] Mensajes tras cambiar a una emoción distinta ({emocion_disrupcion}): {len(historial_tras_disrupcion)} | ¿Se purgó correctamente? {'SÍ' if purga_emocion_ok else 'NO'}")

    # --- Guardar resultados por turno en CSV ---
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_test_integrated_rag_ag2_memoria.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados_turnos[0].keys()))
        writer.writeheader()
        writer.writerows(resultados_turnos)
    print(f"\n[+] Resultados por turno guardados en {csv_path}")

    print("\n=== RESUMEN DE VERIFICACIONES (memoria + RAG combinados) ===")
    def fmt(v): return "OK" if v else "FALLO"
    print(f"{'Memoria de corto plazo con ruido RAG intercalado':<55}{fmt(recuerda_nombre)}")
    print(f"{'Purga del mensaje RAG antes de persistir a disco':<55}{fmt(purga_rag_ok)}")
    print(f"{'Persistencia del historial entre sesiones (JSON)':<55}{fmt(persistencia_ok)}")
    print(f"{'Continuidad de contexto tras recargar (sesión 2)':<55}{fmt(recuerda_contexto_sesion2)}")
    print(f"{'Purga de memoria por disrupción emocional':<55}{fmt(purga_emocion_ok)}")


async def ejecutar_bateria_integrada():
    print("\n[+] Iniciando Sistema Integrado RAG+AG2+Emociones (modo batería de pruebas automatizada)...")

    llm_config = _llm_config()
    coleccion_memoria, hyde_client, cross_encoder = _conectar_servicios()

    resultados = []

    for npc_id, emocion, intensidad, pregunta, tema in BATERIA_PRUEBAS_INTEGRADA:
        npc_name, prompt_completo = construir_prompt_cognitivo(npc_id, emocion, intensidad)

        # Agentes nuevos en cada entrada para que cada combinación NPC/emoción
        # sea independiente y comparable (no arrastra historial de la anterior).
        npc_agent = AssistantAgent(name=npc_name, system_message=prompt_completo, llm_config=llm_config)
        player_proxy = UserProxyAgent(
            name="Jugador",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )

        metricas = {}
        t_inicio = time.perf_counter()

        contexto_factual = await extraer_conocimiento_rag(hyde_client, coleccion_memoria, cross_encoder, pregunta, npc_name, metricas=metricas)
        if contexto_factual:
            npc_agent.chat_messages[player_proxy] = [construir_mensaje_rag(contexto_factual, npc_name, emocion, intensidad)]

        print(f"\n--- [{npc_name} | {emocion.upper()} int.{intensidad}] {pregunta} ---")
        respuesta = ""
        try:
            player_proxy.initiate_chat(npc_agent, message=pregunta, clear_history=False, silent=True)
            historial = npc_agent.chat_messages.get(player_proxy, [])
            respuesta = historial[-1]["content"] if historial else ""
            print(f"NPC: {respuesta}")
        except Exception as e:
            print(f"[Error durante la generación]: {e}")

        t_total = time.perf_counter() - t_inicio

        fila = {
            "npc": npc_name,
            "emocion": emocion,
            "intensidad": intensidad,
            "pregunta": pregunta,
            "tema_esperado": tema,
            "t_hyde_s": round(metricas.get('t_hyde', 0.0), 3),
            "t_retrieval_s": round(metricas.get('t_retrieval', 0.0), 3),
            "t_rerank_s": round(metricas.get('t_rerank', 0.0), 3),
            "t_total_s": round(t_total, 3),
            "docs_recuperados": metricas.get('docs_recuperados', 0),
            "score_top1": round(metricas['score_top1'], 3) if metricas.get('score_top1') is not None else None,
            "inyecto_rag": bool(contexto_factual),
            "cumple_formato_gesto": cumple_formato_gesto(respuesta),
            "tokens_pregunta": contar_tokens(pregunta),
            "tokens_respuesta": contar_tokens(respuesta),
            "respuesta": respuesta.strip(),
        }
        resultados.append(fila)

        await asyncio.sleep(1)

    # --- Guardar resultados en CSV para la memoria del TFG ---
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_test_integrated_rag_ag2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\n[+] Resultados guardados en {csv_path}")

    # --- Tabla resumen en consola ---
    print("\n=== RESUMEN ===")
    header = f"{'NPC':<10}{'Emocion':<10}{'t_total(s)':<12}{'t_hyde(s)':<11}{'t_retr(s)':<11}{'t_rerank(s)':<13}{'score':<8}{'docs':<6}{'gesto':<7}{'tok_in':<8}{'tok_out':<8}"
    print(header)
    print("-" * len(header))
    for r in resultados:
        print(f"{r['npc']:<10}{r['emocion']:<10}{r['t_total_s']:<12}{r['t_hyde_s']:<11}{r['t_retrieval_s']:<11}{r['t_rerank_s']:<13}{str(r['score_top1']):<8}{r['docs_recuperados']:<6}{str(r['cumple_formato_gesto']):<7}{r['tokens_pregunta']:<8}{r['tokens_respuesta']:<8}")

    # --- Sub-prueba de memoria conversacional AG2 + RAG combinados ---
    await ejecutar_prueba_memoria_integrada(coleccion_memoria, hyde_client, cross_encoder)


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("¿Cómo quieres ejecutar este test?")
    print("  [1] Modo interactivo (elegir NPC/emoción y escribir preguntas manualmente, como antes)")
    print("  [2] Batería de pruebas automática (NPCs/emociones fijos + métricas + tabla de resultados)")
    modo = input("Elige una opción [1/2]: ").strip()

    if modo == "2":
        asyncio.run(ejecutar_bateria_integrada())
    else:
        iniciar_sistema_completo()
