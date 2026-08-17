import torch
import asyncio
import sys
import os
import traceback
import logging
import subprocess # <-- 1. Añadimos esta importación nativa

from src.http.http_server import http_server
from src.http.routes.routeable import routeable
from src.http.routes.mantella_route import mantella_route
from src.setup import MantellaSetup
from src.ui.start_ui import StartUI

# --- Importaciones de la Arquitectura Integrada (TFG) ---
import chromadb
from chromadb.utils import embedding_functions
# Importamos tu gestor por si es necesario precargar o limpiar memoria al inicio
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2 

def main():
    # 1. Resolución de concurrencia para Windows
    # Fundamental para evitar bloqueos del event loop con llamadas asíncronas de AG2/HyDE
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        config, language_info = MantellaSetup().initialise(
            config_file='config.ini',
            logging_file='logging.log', 
            language_file='data/language_support.csv')

        # Pequeño cambio para identificar que estás corriendo tu versión modificada
        mantella_version = '0.13.1 - TFG AG2 Edition'
        logging.log(24, f'\nMantella v{mantella_version}')

        # 2. Inicialización de la Base de Conocimiento (ChromaDB)
        logging.info("[Main] Inicializando motor de base de datos vectorial local (ChromaDB)...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "npc_chroma_db")
        
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        chroma_client = chromadb.PersistentClient(path=db_path)
        
        coleccion_memoria = chroma_client.get_or_create_collection(
            name="memoria_global_npcs", 
            embedding_function=emb_fn
        )
        logging.info("[Main] Memoria global de NPCs cargada y lista.")

        # --- CORRECCIÓN: Instanciar tu gestor RAG y empaquetar la memoria ---
        # Asegúrate de que GestorMemoriaAG2 reciba la colección en su constructor
        gestor_rag_ag2 = GestorMemoriaAG2(coleccion_memoria)

        # 3. Configuración del Servidor HTTP
        mantella_http_server = http_server()
        should_debug_http = config.show_http_debug_messages
        
        # 4. Enrutamiento Principal
        # --- CORRECCIÓN: Inyectar la dependencia de memoria al router ---
        conversation = mantella_route(
            config=config, 
            stt_secret_key_file='STT_SECRET_KEY.txt', 
            image_secret_key_file='IMAGE_SECRET_KEY.txt', 
            function_llm_secret_key_file='FUNCTION_GPT_SECRET_KEY.txt',
            secret_key_file='GPT_SECRET_KEY.txt', 
            language_info=language_info, 
            show_debug_messages=should_debug_http,
            # Añade este parámetro en la definición de mantella_route para que AG2 lo use
            gestor_memoria=gestor_rag_ag2 
        )
        
        # Precargamos aqui todo lo pesado (CrossEncoder, TTS headless, ChromaDB) para que
        # ocurra durante el arranque visible y no en la primera peticion real del juego.
        # Si no se hace, el juego agota su propio timeout esperando la primera respuesta
        # y cancela la conversacion antes de que el backend termine de cargar.
        logging.info("[Main] Precargando modelos pesados antes de aceptar conexiones del juego...")
        try:
            conversation._can_route_be_used()
            logging.info("[Main] Precarga completa. El backend puede responder de inmediato.")
        except Exception:
            logging.error("[Main] La precarga de modelos pesados ha fallado:")
            logging.error(traceback.format_exc())
            raise

        ui = StartUI(config)
        routes: list[routeable] = [conversation, ui]
        
        



        

        # 5. Arranque del sistema
        logging.info("[Main] Levantando servidor HTTP para Skyrim. Esperando peticiones del motor...")
        mantella_http_server.start(int(config.port), routes, config.play_startup_sound, should_debug_http)

    except Exception as e:
        logging.error("".join(traceback.format_exception(e)))
        input("Press Enter to exit.")

if __name__ == '__main__':
    main()