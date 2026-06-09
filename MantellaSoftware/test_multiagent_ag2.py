import asyncio
import sys
import os
import logging
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IMPORTAMOS LAS CLASES DE GRUPO DE AUTOGEN
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from src.llm.client_base import ClientBase
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

from src.llm.npc_emotions import REGISTRO_NPCS, EMOCIONES_EKMAN, construir_prompt_cognitivo

async def extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, texto_usuario: str) -> str:
    """
    Recupera contexto factual de ChromaDB a nivel global (sin filtrar por un NPC específico)
    para alimentar el debate del grupo.
    """
    try:
        prompt_hyde = (
            f"Escribe un párrafo enciclopédico, técnico y muy breve que responda directamente "
            f"a la siguiente consulta: '{texto_usuario}'. "
            f"Proporciona datos fácticos. NUNCA uses lenguaje de rol."
        )
        
        response = await hyde_client.chat.completions.create(
            model="gemini-flash-latest",
            messages=[{"role": "user", "content": prompt_hyde}],
            temperature=0.1, 
            max_tokens=120
        )
        doc_hipotetico = response.choices[0].message.content.strip()

        # En modo multiagente buscamos en toda la BBDD sin filtrar por npc_name
        resultados = coleccion_memoria.query(
            query_texts=[doc_hipotetico], 
            n_results=3
        )
        
        if resultados and resultados['documents'] and len(resultados['documents'][0]) > 0:
            return "\n\n".join(resultados['documents'][0])
            
        return ""
    except Exception as e:
        logging.error(f"[RAG] Error en la extracción de conocimiento: {e}")
        return ""

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

        npc_name, npc_prompt_completo = construir_prompt_cognitivo(seleccion, emocion_elegida, intensidad)
        
        # Añadimos una regla extra vital para los GroupChats: No hablar por los demás
        prompt_grupal = npc_prompt_completo + (
            "\n\nREGLA DE DEBATE GRUPAL: Estás en una sala con otros personajes. "
            "DEBES referirte a ellos por su nombre si les contestas. NUNCA generes respuestas ni hables en nombre de los demás."
        )

        nombres_elegidos.append(npc_name)
        agentes_npc_instanciados.append({
            "name": npc_name,
            "prompt": prompt_grupal
        })

    # ==========================================
    # INICIALIZACIÓN DE SERVICIOS Y RAG
    # ==========================================
    api_key = ClientBase._get_api_key(["GPT_SECRET_KEY.txt"], show_error=False)
    if not api_key:
        print("[!] ERROR: No se encontró GPT_SECRET_KEY.txt.")
        return

    llm_config = {
        "config_list": [{"model": "gemini-flash-latest", 
        "api_key": api_key, 
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", 
        "price": [0.0, 0.0]}],
        "temperature": 0.7,
    }

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "npc_chroma_db"))
    coleccion_memoria = chroma_client.get_or_create_collection(
        name="memoria_global_npcs", embedding_function=emb_fn
    )
    hyde_client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai")

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
        speaker_selection_method="auto"
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
    # TEXTO DE REGLAS ESTRICTAS PARA MULTIAGENTE
    reglas_rag_grupo = (
        "\n\n[REGLAS CRÍTICAS DEL SISTEMA PARA EL GRUPO]:\n"
        "1. FILTRO DE IDENTIDAD: Si decidís usar esta información, hacedlo STRICTAMENTE bajo vuestra personalidad y estado emocional actual.\n"
        "2. REGLA DE IGNORANCIA: Si esta información no tiene sentido para vosotros o no responde al tema, ignoradla por completo."
    )

    if modo == "1":
        print("\n[Modo Observador Activo]")
        tema_inicial = input("Introduce un evento o frase inicial para detonar la conversación entre ellos: ")
        
        # FIX 1: Extracción RAG en Modo Observador
        contexto_factual = loop.run_until_complete(
            extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, tema_inicial)
        )
        
        mensaje_final = tema_inicial
        if contexto_factual:
            print(f"[*] Recuperando contexto del mundo para el debate...")
            mensaje_final = f"CONTEXTO FACTUAL: {contexto_factual}{reglas_rag_grupo}\n\nDetonante de la charla: {tema_inicial}"

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
                    extraer_conocimiento_rag_global(hyde_client, coleccion_memoria, texto_usuario)
                )

                mensaje_final = texto_usuario
                if contexto_factual:
                    print(f"[*] Recuperando contexto del mundo para el debate...")
                    # FIX 2 y 3: Inyección controlada con reglas de identidad
                    mensaje_final = (
                        f"[SISTEMA: La siguiente información contextualiza la duda del jugador. {reglas_rag_grupo}]\n"
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
                
                # FIX 3 (Mitigación de Bloat): Limpiamos la inyección del sistema del último mensaje
                # para que los NPCs recuerden lo que dijo el jugador, pero no acumulen la misma info de la BBDD
                if contexto_factual and groupchat.messages:
                    ultimo_msg = groupchat.messages[-1]
                    if "El Jugador dice:" in ultimo_msg.get("content", ""):
                        ultimo_msg["content"] = f"El Jugador dijo: {texto_usuario}"
                
                primer_mensaje = False
                
        except KeyboardInterrupt:
            print("\n[!] Cierre forzado detectado (Ctrl+C).")
        finally:
            print(f"\n[+] Sesión de debate cerrada con seguridad.")
            # Opcional: Aquí deberías iterar sobre `agentes_npc_instanciados` e invocar tu GestorMemoriaAG2 si deseas guardar.

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    iniciar_sistema_multiagente()