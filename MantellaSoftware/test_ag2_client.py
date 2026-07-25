import asyncio
import sys
import os
import logging

# Configuración básica
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen import AssistantAgent, UserProxyAgent
from src.llm.client_base import ClientBase
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

def iniciar_consola_ag2_1vs1():
    print("\n[+] Iniciando AG2 (1 vs 1) con MEMORIA PERSISTENTE...")

    # 1. API Key
    api_key = ClientBase._get_api_key(["GPT_SECRET_KEY.txt"], show_error=False)
    if not api_key:
        print("[!] ERROR: No se encontró GPT_SECRET_KEY.txt.")
        return

    llm_config = {
        "config_list": [{"model": "gemini-flash-latest", 
        "api_key": api_key, 
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "price": [0.0, 0.0]
        }],
        "temperature": 0.8,
        "timeout": 120,
    }

    # 2. Cargar memoria
    nombre_npc = "Farengar"
    gestor = GestorMemoriaAG2()
    historial_previo = gestor.cargar_conversacion(nombre_npc, "neutral")

    # 3. Crear agentes
    farengar_agent = AssistantAgent(
        name=nombre_npc,
        system_message="Eres Farengar Fuego Secreto, mago de Carrera Blanca. TUS RESPUESTAS DEBEN SER BREVES.",
        llm_config=llm_config,
    )

    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER", 
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # 4. Inyectar memoria anterior
    if historial_previo:
        print(f"[*] Cargando {len(historial_previo)} mensajes anteriores...")
        # Inyectamos el historial directamente en la memoria del agente
        farengar_agent.chat_messages[player_proxy] = historial_previo

    print("\n=== CONSOLA LISTA ===")
    
    try:
        while True:
            texto_usuario = input("\nTú: ")
            if texto_usuario.lower() in ['salir', 'exit', 'quit']:
                break
            
            # Lanzar el chat (silent=False para que veas la respuesta)
            player_proxy.initiate_chat(
                farengar_agent,
                message=texto_usuario,
                clear_history=False,
                silent=False 
            )
            
    finally:
        # 5. Guardar al salir
        historial_actualizado = farengar_agent.chat_messages.get(player_proxy, [])
        gestor.guardar_conversacion(nombre_npc, historial_actualizado)
        print(f"\n[+] Memoria guardada ({len(historial_actualizado)} mensajes).")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    iniciar_consola_ag2_1vs1()