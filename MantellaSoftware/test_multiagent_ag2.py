import asyncio
import sys
import os
import re
import time
import csv
import logging
from collections import Counter
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import numpy as np
import tiktoken

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2
from src.llm.npc_emotions import REGISTRO_NPCS, EMOCIONES_EKMAN, construir_prompt_cognitivo

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_MODEL = "gemma4:e2b"

_encoding = tiktoken.get_encoding("cl100k_base")

def contar_tokens(texto: str) -> int:
    return len(_encoding.encode(texto or ""))


def cumple_formato_gesto(respuesta: str) -> bool:
    """Comprueba que el mensaje empiece por un único gesto físico entre corchetes."""
    return bool(re.match(r'^\s*\[[^\[\]]+\]', respuesta or ""))


async def extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, cross_encoder: CrossEncoder, texto_usuario: str, metricas: dict | None = None) -> str:
    """
    Recupera contexto factual de ChromaDB a nivel global (sin filtrar por un NPC
    específico) para alimentar el debate del grupo, con HyDE + re-ranking por
    cross-encoder (igual que test_integrated_rag_ag2.py / MyCustomClient).
    """
    try:
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

        # Búsqueda global (sin filtrar por npc_name) sobre hasta 15 candidatos
        t1 = time.perf_counter()
        total_docs = coleccion_memoria.count()
        resultados = coleccion_memoria.query(
            query_texts=[doc_hipotetico],
            n_results=min(15, total_docs) if total_docs else 0,
        )
        if metricas is not None:
            metricas['t_retrieval'] = time.perf_counter() - t1

        candidatos = resultados['documents'][0] if resultados and resultados.get('documents') and resultados['documents'][0] else []
        if metricas is not None:
            metricas['docs_recuperados'] = len(candidatos)

        if not candidatos:
            return ""

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


REGLAS_RAG_GRUPO = (
    "\n\n[REGLAS CRÍTICAS DEL SISTEMA PARA EL GRUPO]:\n"
    "0. USO DEL CONOCIMIENTO: Si el CONTEXTO FACTUAL SÍ responde al tema, asumidlo como conocimiento propio y usadlo "
    "en vuestra respuesta (no digáis que no tenéis información sobre ello).\n"
    "1. FILTRO DE IDENTIDAD: Si decidís usar esta información, hacedlo STRICTAMENTE bajo vuestra personalidad y estado emocional actual.\n"
    "2. REGLA DE IGNORANCIA: Si esta información no tiene sentido para vosotros o no responde al tema, ignoradla por completo."
)


def _prompt_base(npc_id: str, emocion: str, intensidad: int) -> tuple[str, str]:
    """Personalidad + emoción del NPC, sin la regla de debate grupal todavía."""
    return construir_prompt_cognitivo(npc_id, emocion, intensidad)


def _regla_debate_grupal(otros_nombres: list[str]) -> str:
    """Regla de comportamiento en grupo, con un ejemplo concreto usando el
    nombre real de otro participante."""
    if not otros_nombres:
        return ""
    ejemplo = otros_nombres[0]
    return (
        "\n\nREGLA DE DEBATE GRUPAL (OBLIGATORIA, no opcional): Estás en una sala con estos otros "
        f"personajes: {', '.join(otros_nombres)}. Tu respuesta es INVÁLIDA si no incluye el nombre de "
        f"la persona a la que respondes. FORMATO EXACTO A SEGUIR: '[gesto] {ejemplo}, <tu réplica>'. "
        f"Por ejemplo: '[gesto] {ejemplo}, eso que dices es absurdo.' "
        "Empieza siempre tu diálogo (justo después del gesto) nombrando a quien te habló. "
        "NUNCA generes respuestas ni hables en nombre de los demás."
    )


def _conectar_servicios():
    """Inicializa ChromaDB + cliente HyDE + cross-encoder de re-ranking, igual
    que en test_integrated_rag_ag2.py."""
    print("[Base de Conocimiento] Conectando a ChromaDB local...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "npc_chroma_db"))
    coleccion_memoria = chroma_client.get_or_create_collection(
        name="memoria_global_npcs", embedding_function=emb_fn
    )

    hyde_client = AsyncOpenAI(api_key="abc123", base_url=OLLAMA_BASE_URL)

    print("[Re-Ranker] Cargando modelo CrossEncoder en VRAM (CUDA)...")
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cuda')

    return coleccion_memoria, hyde_client, cross_encoder


def _llm_config() -> dict:
    return {
        "config_list": [{
            "model": OLLAMA_MODEL,
            "api_key": "abc123",
            "base_url": OLLAMA_BASE_URL,
            "price": [0.0, 0.0],
        }],
        "temperature": 0.7,
        "cache_seed": None,
    }


def iniciar_sistema_multiagente():
    print("\n[+] Iniciando Ecosistema Multiagente (GroupChat + Ekman)... ")

    # ==========================================
    # CONFIGURACIÓN DEL ENTORNO DE SIMULACIÓN
    # ==========================================
    print("\n=== MODO DE SIMULACIÓN ===")
    print("1. Modo Observador (Solo los NPCs debaten entre ellos sobre un tema inicial)")
    print("2. Modo Interactivo (El Jugador participa libremente en la conversación)")

    modo = ""
    while modo not in ["1", "2"]:
        modo = input("Elige el modo de simulación (1 o 2): ")

    num_npcs = 0
    while num_npcs < 2 or num_npcs > len(REGISTRO_NPCS):
        try:
            num_npcs = int(input(f"\n¿Cuántos NPCs quieres en la conversación? (Mínimo 2, Máximo {len(REGISTRO_NPCS)}): "))
        except ValueError:
            continue

    agentes_npc_instanciados = []
    nombres_elegidos = []

    # ==========================================
    # CREACIÓN DINÁMICA DE LA SALA DE DEBATE
    # ==========================================
    for i in range(num_npcs):
        print(f"\n--- CONFIGURANDO NPC {i+1} ---")
        for key, data in REGISTRO_NPCS.items():
            if data['nombre'] not in nombres_elegidos:
                print(f"{key}. {data['nombre']}")

        seleccion = ""
        while seleccion not in REGISTRO_NPCS or REGISTRO_NPCS[seleccion]['nombre'] in nombres_elegidos:
            seleccion = input("Elige el número del NPC: ")
            if REGISTRO_NPCS[seleccion]['nombre'] in nombres_elegidos:
                print("Ese NPC ya está en la sala. Elige otro.")

        emociones_disponibles = list(EMOCIONES_EKMAN.keys())
        for idx, emo in enumerate(emociones_disponibles, 1):
            print(f"{idx}. {emo.capitalize()}")

        sel_emo = -1
        while sel_emo not in range(1, len(emociones_disponibles) + 1):
            try:
                sel_emo = int(input("Elige el estado emocional (1-8): "))
            except ValueError:
                continue

        emocion_elegida = emociones_disponibles[sel_emo - 1]

        intensidad = 3
        if emocion_elegida != "neutral":
            while True:
                try:
                    intensidad = int(input("Intensidad emocional (1-5): "))
                    if 1 <= intensidad <= 5: break
                except ValueError:
                    continue

        npc_name, prompt_base = _prompt_base(seleccion, emocion_elegida, intensidad)

        nombres_elegidos.append(npc_name)
        agentes_npc_instanciados.append({
            "name": npc_name,
            "prompt": prompt_base
        })

    # Añadimos la regla de debate grupal en una segunda pasada, ya con el
    # reparto completo de la sala conocido, para poder dar un ejemplo con el
    # nombre real de al menos otro participante.
    for npc_data in agentes_npc_instanciados:
        otros = [n for n in nombres_elegidos if n != npc_data["name"]]
        npc_data["prompt"] += _regla_debate_grupal(otros)

    # ==========================================
    # INICIALIZACIÓN DE SERVICIOS Y RAG
    # ==========================================
    llm_config = _llm_config()
    coleccion_memoria, hyde_client, cross_encoder = _conectar_servicios()

    # ==========================================
    # CONSTRUCCIÓN DEL GROUPCHAT AG2
    # ==========================================
    agentes_autogen = []
    for npc_data in agentes_npc_instanciados:
        agente = AssistantAgent(
            name=npc_data["name"],
            system_message=npc_data["prompt"],
            llm_config=llm_config,
        )
        agentes_autogen.append(agente)

    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    lista_participantes = agentes_autogen.copy()
    if modo == "2":
        lista_participantes.append(player_proxy)

    groupchat = GroupChat(
        agents=lista_participantes,
        messages=[],
        max_round=15, # Límite de seguridad para que no hablen eternamente solos
        # AutoGen avisa de que "auto" con pocos agentes tiende a repetir
        # siempre al mismo hablante ("GroupChat is underpopulated"); con
        # round_robin el turno se reparte de forma predecible y equilibrada
        # entre todos los participantes, independientemente de cuántos haya.
        speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    print(f"\n=== SALA DE DEBATE LISTA ===")
    print(f"Participantes: {', '.join(nombres_elegidos)}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # ==========================================
    # EJECUCIÓN DEL MODO SELECCIONADO
    # ==========================================
    if modo == "1":
        print("\n[Modo Observador Activo]")
        tema_inicial = input("Introduce un evento o frase inicial para detonar la conversación entre ellos: ")

        contexto_factual = loop.run_until_complete(
            extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, cross_encoder, tema_inicial)
        )

        mensaje_final = tema_inicial
        if contexto_factual:
            print(f"[*] Recuperando contexto del mundo para el debate...")
            mensaje_final = f"CONTEXTO FACTUAL: {contexto_factual}{REGLAS_RAG_GRUPO}\n\nDetonante de la charla: {tema_inicial}"

        print("\n--- INICIO DEL DEBATE ---")
        player_proxy.initiate_chat(manager, message=mensaje_final, clear_history=True)
        print("\n[+] Debate finalizado por límite de turnos o consenso natural.")

    else:
        print("Escribe 'salir' para terminar.\n")
        try:
            primer_mensaje = True
            while True:
                texto_usuario = input("\nTú: ")
                if texto_usuario.lower() in ['salir', 'exit', 'quit']:
                    break
                if not texto_usuario.strip(): continue

                contexto_factual = loop.run_until_complete(
                    extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, cross_encoder, texto_usuario)
                )

                mensaje_final = texto_usuario
                if contexto_factual:
                    print(f"[*] Recuperando contexto del mundo para el debate...")
                    mensaje_final = (
                        f"[SISTEMA: La siguiente información contextualiza la duda del jugador. {REGLAS_RAG_GRUPO}]\n"
                        f"INFORMACIÓN: {contexto_factual}\n"
                        f"----------------------------------------\n"
                        f"El Jugador dice: {texto_usuario}"
                    )

                player_proxy.initiate_chat(
                    manager,
                    message=mensaje_final,
                    clear_history=primer_mensaje,
                    silent=False
                )

                # Limpiamos la inyección del sistema del último mensaje para que los
                # NPCs recuerden lo que dijo el jugador, pero no acumulen la misma
                # info de la BBDD en cada turno.
                if contexto_factual and groupchat.messages:
                    ultimo_msg = groupchat.messages[-1]
                    if "El Jugador dice:" in ultimo_msg.get("content", ""):
                        ultimo_msg["content"] = f"El Jugador dijo: {texto_usuario}"

                primer_mensaje = False

        except KeyboardInterrupt:
            print("\n[!] Cierre forzado detectado (Ctrl+C).")
        finally:
            print(f"\n[+] Sesión de debate cerrada con seguridad.")


GRUPO_BATERIA = [("1", "alegria", 2), ("2", "alegria", 2)]  # Farengar, Nazeem
TEMA_BATERIA = "¿Quién es Tony Stark?"
# No incluye "stark" (ya aparece en TEMA_BATERIA): solo cuentan palabras que
# únicamente pueden venir del documento RAG inyectado.
TEMA_KEYWORDS = ["iron man", "armadura", "multimillonario", "genio", "fabricante de armas", "electromec"]
MAX_ROUND_BATERIA = 8


async def ejecutar_bateria_multiagente():
    print("\n[+] Iniciando Ecosistema Multiagente (modo batería de pruebas automatizada)...")

    llm_config = _llm_config()
    coleccion_memoria, hyde_client, cross_encoder = _conectar_servicios()

    agentes_npc_instanciados = []
    for npc_id, emocion, intensidad in GRUPO_BATERIA:
        npc_name, prompt_base = _prompt_base(npc_id, emocion, intensidad)
        agentes_npc_instanciados.append({"name": npc_name, "prompt": prompt_base, "emocion": emocion, "intensidad": intensidad})

    nombres_npc = [a["name"] for a in agentes_npc_instanciados]

    agentes_autogen = []
    for npc_data in agentes_npc_instanciados:
        otros = [n for n in nombres_npc if n != npc_data["name"]]
        prompt_grupal = npc_data["prompt"] + _regla_debate_grupal(otros)
        agentes_autogen.append(AssistantAgent(name=npc_data["name"], system_message=prompt_grupal, llm_config=llm_config))

    metricas = {}
    t_inicio = time.perf_counter()

    contexto_factual = await extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, cross_encoder, TEMA_BATERIA, metricas=metricas)
    mensaje_final = TEMA_BATERIA
    if contexto_factual:
        mensaje_final = f"CONTEXTO FACTUAL: {contexto_factual}{REGLAS_RAG_GRUPO}\n\nDetonante de la charla: {TEMA_BATERIA}"

    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    groupchat = GroupChat(
        agents=agentes_autogen,
        messages=[],
        max_round=MAX_ROUND_BATERIA,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    print(f"\n--- [Debate: {', '.join(nombres_npc)}] {TEMA_BATERIA} ---")
    try:
        player_proxy.initiate_chat(manager, message=mensaje_final, clear_history=True, silent=True)
    except Exception as e:
        print(f"[Error durante el debate]: {e}")

    t_total = time.perf_counter() - t_inicio

    # --- Análisis por mensaje ---
    mensajes_npc = [m for m in groupchat.messages if m.get("name") in nombres_npc]

    filas_mensajes = []
    for idx, m in enumerate(mensajes_npc, start=1):
        hablante = m.get("name", "")
        contenido = m.get("content", "") or ""
        otros = [n for n in nombres_npc if n != hablante]
        menciona_otro = any(re.search(rf'\b{re.escape(n)}\b', contenido, re.IGNORECASE) for n in otros)
        filas_mensajes.append({
            "turno": idx,
            "hablante": hablante,
            "contenido": contenido.strip(),
            "cumple_formato_gesto": cumple_formato_gesto(contenido),
            "menciona_a_otro_npc": menciona_otro,
            "tokens": contar_tokens(contenido),
        })
        print(f"\n[{hablante}]: {contenido}")

    speakers_unicos = sorted(set(m["hablante"] for m in filas_mensajes))
    mensajes_por_npc = Counter(m["hablante"] for m in filas_mensajes)
    pct_gesto = sum(1 for m in filas_mensajes if m["cumple_formato_gesto"]) / len(filas_mensajes) if filas_mensajes else 0.0
    pct_menciona_otro = sum(1 for m in filas_mensajes if m["menciona_a_otro_npc"]) / len(filas_mensajes) if filas_mensajes else 0.0

    texto_completo = " ".join(m["contenido"].lower() for m in filas_mensajes)
    uso_contexto_rag = any(re.search(rf'\b{re.escape(kw)}\b', texto_completo) for kw in TEMA_KEYWORDS)

    print("\n[Verificación] ¿Participaron los NPCs esperados?", "SÍ" if set(speakers_unicos) == set(nombres_npc) else f"NO (hablaron: {speakers_unicos})")
    print(f"[Verificación] % de mensajes que cumplen el formato de gesto: {pct_gesto*100:.0f}%")
    print(f"[Verificación] % de mensajes que se refieren a otro NPC por su nombre: {pct_menciona_otro*100:.0f}%")
    print(f"[Verificación] ¿El grupo usó el contexto RAG inyectado (menciona Stark/armadura/etc.)? {'SÍ' if uso_contexto_rag else 'NO'}")

    # --- CSV 1: resumen del debate ---
    resumen = {
        "npcs": ", ".join(nombres_npc),
        "tema": TEMA_BATERIA,
        "t_hyde_s": round(metricas.get('t_hyde', 0.0), 3),
        "t_retrieval_s": round(metricas.get('t_retrieval', 0.0), 3),
        "t_rerank_s": round(metricas.get('t_rerank', 0.0), 3),
        "t_total_s": round(t_total, 3),
        "docs_recuperados": metricas.get('docs_recuperados', 0),
        "score_top1": round(metricas['score_top1'], 3) if metricas.get('score_top1') is not None else None,
        "total_mensajes_npc": len(filas_mensajes),
        "speakers_unicos": ", ".join(speakers_unicos),
        "mensajes_por_npc": dict(mensajes_por_npc),
        "pct_cumple_gesto": round(pct_gesto, 3),
        "pct_menciona_otro_npc": round(pct_menciona_otro, 3),
        "uso_contexto_rag": uso_contexto_rag,
    }
    csv_resumen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_test_multiagent_ag2.csv")
    with open(csv_resumen, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen.keys()))
        writer.writeheader()
        writer.writerow(resumen)
    print(f"\n[+] Resumen del debate guardado en {csv_resumen}")

    # --- CSV 2: desglose por mensaje ---
    if filas_mensajes:
        csv_mensajes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_test_multiagent_ag2_mensajes.csv")
        with open(csv_mensajes, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(filas_mensajes[0].keys()))
            writer.writeheader()
            writer.writerows(filas_mensajes)
        print(f"[+] Desglose por mensaje guardado en {csv_mensajes}")

    print("\n=== RESUMEN ===")
    print(f"NPCs: {', '.join(nombres_npc)} | Turnos NPC: {len(filas_mensajes)} | t_total: {t_total:.3f}s")
    print(f"Reparto de turnos: {dict(mensajes_por_npc)}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("¿Cómo quieres ejecutar este test?")
    print("  [1] Modo interactivo (configurar sala de debate manualmente, como antes)")
    print("  [2] Batería de pruebas automática (2 NPCs fijos + métricas + tabla de resultados)")
    modo_ejecucion = input("Elige una opción [1/2]: ").strip()

    if modo_ejecucion == "2":
        asyncio.run(ejecutar_bateria_multiagente())
    else:
        iniciar_sistema_multiagente()
