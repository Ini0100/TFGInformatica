import asyncio
import sys
import os
import logging
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen import AssistantAgent, UserProxyAgent
from src.llm.client_base import ClientBase
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

# IMPORTAMOS NUESTRA NUEVA FACTORÍA EMOCIONAL
from src.llm.npc_emotions import REGISTRO_NPCS, EMOCIONES_EKMAN, construir_prompt_cognitivo

async def extraer_conocimiento_rag(hyde_client, coleccion_memoria, texto_usuario: str, npc_name: str) -> str:
    """
    Recupera contexto factual de ChromaDB usando la técnica HyDE para la fase de pruebas.
    """
    try:
        # 1. Generación del documento hipotético (HyDE)
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

        # 2. Búsqueda en ChromaDB
        resultados = coleccion_memoria.query(
            query_texts=[doc_hipotetico], 
            n_results=3, # Traemos los 3 mejores resultados directamente para el test
            where={"npc_name": npc_name} if npc_name != "Desconocido" else None
        )
        
        if resultados and resultados['documents'] and len(resultados['documents'][0]) > 0:
            return "\n\n".join(resultados['documents'][0])
            
        return ""
        
    except Exception as e:
        logging.error(f"[RAG] Error en la extracción de conocimiento: {e}")
        return ""

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
    api_key = ClientBase._get_api_key(["GPT_SECRET_KEY.txt"], show_error=False)
    if not api_key:
        print("[!] ERROR: No se encontró GPT_SECRET_KEY.txt.")
        return

    llm_config = {
        "config_list": [{
            "model": "gemini-flash-latest", 
            "api_key": api_key, 
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "price": [0.0, 0.0]  # <--- Añade esta línea para silenciar el aviso
        }],
        "temperature": 0.7,
    }

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
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai"
    )

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
                extraer_conocimiento_rag(hyde_client, coleccion_memoria, texto_usuario, npc_name)
            )

            # ¡CORREGIDO! Alineado con 12 espacios
            if contexto_factual:
                print(f"[*] Inyectando conocimiento externo en la memoria de {npc_name}...")

                # --- AÑADE ESTO PARA DEPURAR ---
                print("\n" + "="*40)
                print("🔍 [DEBUG RAG] TEXTO RECUPERADO DE CHROMADB:")
                print(contexto_factual)
                print("="*40 + "\n")
                # -------------------------------
                
                # NUEVO PROMPT ESTRICTO DINÁMICO (Equilibrado)
                prompt_estricto = (
                    "ATENCIÓN AL SISTEMA (DIRECTRIZ DE MÁXIMA PRIORIDAD):\n"
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
                msg_rag = {"role": "system", "content": prompt_estricto}
                npc_agent.chat_messages[player_proxy].append(msg_rag)

            player_proxy.initiate_chat(
                npc_agent,
                message=texto_usuario,
                clear_history=False,
                silent=False
            )
            
    # Solo un bloque de KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n[!] Cierre forzado detectado (Ctrl+C). Iniciando guardado de emergencia...")
        
    finally:
        # 1. Filtramos primero el ruido de las inyecciones de contexto del RAG
        historial_final = [
            m for m in npc_agent.chat_messages.get(player_proxy, []) 
            if "ATENCIÓN AL SISTEMA:" not in m.get("content", "")
        ]
        
        # 2. Control de integridad transaccional (Poda de entradas huérfanas)
        # Si el último mensaje del historial purgado tiene el rol 'user', implica
        # que el flujo se cortó antes de que la API registrara la respuesta del NPC.
        if historial_final and historial_final[-1].get("role") == "user":
            print("\n[Memoria] Detectada pregunta huérfana sin respuesta debido a un corte.")
            print("[Memoria] Purgando última entrada del jugador para evitar bucles de repetición...")
            historial_final.pop()  # Eliminamos el mensaje del usuario sin pareja
            
        # 3. Guardamos el historial limpio y balanceado en el disco
        gestor_memoria.guardar_conversacion(npc_name, historial_final, emocion_elegida)
        print(f"\n[+] Sesión guardada limpiamente. ¡Ecosistema RAG+AG2 cerrado con seguridad!")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    iniciar_sistema_completo()