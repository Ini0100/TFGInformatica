import sys
import os
import asyncio
import logging
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions

# Configuración básica
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen import AssistantAgent, UserProxyAgent
from src.llm.client_base import ClientBase
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

async def extraer_conocimiento_rag(hyde_client, coleccion_memoria, texto_usuario: str, npc_name: str) -> str:
    print(f"\n[RAG] Generando documento imaginario (HyDE) para: '{texto_usuario}'...")
    
    # 1. Transformación HyDE directa (Sin pasar por Mantella)
    prompt_hyde = (
        f"Escribe un párrafo enciclopédico, técnico y muy breve que responda a: '{texto_usuario}'. "
        f"Proporciona datos fácticos. NUNCA uses lenguaje de rol."
    )
    response = await hyde_client.chat.completions.create(
        model="gemini-flash-latest",
        messages=[{"role": "user", "content": prompt_hyde}],
        temperature=0.1, 
        max_tokens=120
    )
    texto_busqueda = response.choices[0].message.content.strip()
    
    total_docs = coleccion_memoria.count()
    if total_docs == 0:
        return ""

    # 2. Búsqueda en ChromaDB
    resultados = coleccion_memoria.query(
        query_texts=[texto_busqueda], 
        n_results=min(3, total_docs), 
        where={"npc_name": npc_name}
    )
    
    if not resultados or not resultados['documents'] or not resultados['documents'][0]:
        return ""
        
    return "\n\n".join(resultados['documents'][0])

def iniciar_sistema_completo():
    print("\n[+] Iniciando Sistema Definitivo (Desacoplado del Motor de Skyrim)...")

    # 1. Obtenemos API Key
    api_key = ClientBase._get_api_key(["GPT_SECRET_KEY.txt"], show_error=False)
    if not api_key:
        print("[!] ERROR: No se encontró GPT_SECRET_KEY.txt.")
        return

    llm_config = {
        "config_list": [{"model": "gemini-flash-latest", "api_key": api_key, "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}],
        "temperature": 0.7,
    }

    # 2. LEVANTAMOS RAG NATIVAMENTE (¡Adiós a ConfigFalsa y LLMClient!)
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

    # 3. Cargar la Memoria Episódica
    gestor_memoria = GestorMemoriaAG2()
    npc_name = "Nazeem" 
    historial_previo = gestor_memoria.cargar_conversacion(npc_name)

    # 4. Crear Agentes AG2
    npc_agent = AssistantAgent(
        name=npc_name,
        system_message=f"Eres {npc_name}, un ciudadano arrogante de Skyrim. TUS RESPUESTAS DEBEN SER BREVES.",
        llm_config=llm_config,
    )

    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False # <--- ¡ESTA ES LA LÍNEA MÁGICA!
    )

    if historial_previo:
        print(f"[*] Cargando {len(historial_previo)} recuerdos pasados desde el disco...")
        npc_agent.chat_messages[player_proxy] = historial_previo

    print("\n=== SISTEMA TOTALMENTE INTEGRADO LISTO ===")
    print("Escribe 'salir' para terminar y GUARDAR LA MEMORIA.\n")
    
    loop = asyncio.get_event_loop()

    try:
        while True:
            texto_usuario = input("\nTú: ")
            if texto_usuario.lower() in ['salir', 'exit', 'quit']: 
                break
            if not texto_usuario.strip():
                continue

            # Fase RAG (Llamamos a nuestra función aislada)
            contexto_factual = loop.run_until_complete(
                extraer_conocimiento_rag(hyde_client, coleccion_memoria, texto_usuario, npc_name)
            )

           # Inyección en la memoria de AG2
            if contexto_factual:
                print(f"[*] Inyectando conocimiento externo en la memoria de {npc_name}...")
                prompt_estricto = (
                    "ATENCIÓN AL SISTEMA: Tienes dos fuentes de conocimiento. "
                    "1. Para datos personales, identidades o cosas que habéis hablado antes, CONFÍA EN VUESTRO HISTORIAL DE CHAT.\n"
                    "2. Para preguntas sobre lore, ciencia o datos concretos, usa la siguiente INFORMACIÓN DE REFERENCIA. "
                    "Si le preguntas sobre un tema fáctico que NO está en esta referencia, responde con arrogancia que no pierdes el tiempo en nimiedades.\n\n"
                    f"INFORMACIÓN DE REFERENCIA:\n{contexto_factual}"
                )
                msg_rag = {"role": "system", "content": prompt_estricto}
                npc_agent.chat_messages[player_proxy].append(msg_rag)

            # Fase Conversación
            player_proxy.initiate_chat(
                npc_agent,
                message=texto_usuario,
                clear_history=False,
                silent=False
            )
            
    except KeyboardInterrupt:
        # Esto captura el Ctrl+C y evita que el programa muera sin guardar
        print("\n[!] Cierre forzado detectado (Ctrl+C). Iniciando guardado de emergencia...")
        
    finally:
        # Guardado limpio: filtramos las inyecciones de sistema para que no contaminen la memoria episódica
        historial_final = [
            m for m in npc_agent.chat_messages.get(player_proxy, []) 
            if "ATENCIÓN AL SISTEMA:" not in m.get("content", "")
        ]
        gestor_memoria.guardar_conversacion(npc_name, historial_final)
        print(f"\n[+] Sesión guardada. ¡Ecosistema RAG+AG2 cerrado con seguridad!")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    iniciar_sistema_completo()