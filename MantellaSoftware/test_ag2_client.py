import asyncio
import sys
import os
import logging

# Desactivamos logs molestos de librerías subyacentes
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

# --- EL TRUCO ESTÁ AQUÍ ---
# Como el archivo está en src/ag2/, le decimos a Python que suba dos niveles 
# (..) para llegar a la raíz de MantellaSoftware. Así encontrará todo.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
# --------------------------

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from src.llm.client_base import ClientBase

# --- LÓGICA DE INTERCEPCIÓN (NUEVO) ---
def mantella_print_hook(recipient, messages, sender, config):
    """
    Intercepta los mensajes generados por los agentes ANTES de mostrarlos,
    formateándolos para el ChangeCharacterParser de Mantella.
    """
    if messages:
        last_msg = messages[-1]
        # Solo interceptamos si el mensaje tiene un emisor definido y no es el jugador
        if "name" in last_msg and sender.name != "Jugador":
            speaker_name = last_msg["name"]
            content = last_msg.get("content", "")
            
            # Simulamos el envío a SkyrimSender
            print(f"\n[SKYRIM SENDER INTERCEPT] -> {speaker_name}: {content}")
            
    # Devolver False indica a AG2 que continúe con su flujo normal
    return False, None

def iniciar_consola_ag2():
    print("\n[+] Iniciando el ecosistema Multiagente AG2...")

    # 1. Obtenemos la API Key usando el sistema nativo de Mantella
    api_key = ClientBase._get_api_key(["GPT_SECRET_KEY.txt"], show_error=False)
    if not api_key:
        print("[!] ERROR: No se encontró GPT_SECRET_KEY.txt.")
        return

    llm_config = {
        "config_list": [
            {
                "model": "gemini-flash-latest",
                "api_key": api_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"
            }
        ],
        "temperature": 0.8,
        "timeout": 120,
    }

    # 2. Definimos los agentes NPC
    farengar_agent = AssistantAgent(
        name="Farengar",
        system_message="Eres Farengar Fuego Secreto, el mago de la corte del Jarl en Carrera Blanca. Eres arrogante y estudias dragones. TUS RESPUESTAS DEBEN SER MUY BREVES (1 línea).",
        llm_config=llm_config,
    )
    # Conectamos el hook a Farengar
    farengar_agent.register_reply([None, None], mantella_print_hook, position=1)

    balgruuf_agent = AssistantAgent(
        name="Balgruuf",
        system_message="Eres el Jarl Balgruuf el Mayor. Eres el líder de Carrera Blanca. Te preocupas por tu gente y escuchas a tu mago Farengar. TUS RESPUESTAS DEBEN SER MUY BREVES (1 línea).",
        llm_config=llm_config,
    )
    # Conectamos el hook a Balgruuf
    balgruuf_agent.register_reply([None, None], mantella_print_hook, position=1)

    # 3. Creamos el Agente del Jugador
    player_proxy = UserProxyAgent(
        name="Jugador",
        system_message="Eres el jugador interactuando con los NPCs en Skyrim.",
        human_input_mode="ALWAYS", # Bloqueante para la prueba en consola
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # 4. ORQUESTACIÓN: Creamos el Chat Grupal (NUEVO)
    groupchat = GroupChat(
        agents=[player_proxy, farengar_agent, balgruuf_agent], 
        messages=[], 
        max_round=10,
        speaker_selection_method="auto" # AG2 decide quién habla por contexto
    )

    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config=llm_config
    )

    print("\n=== CONSOLA DE PRUEBAS MANTELLA (MULTIAGENTE) ===")
    print("Escribe 'exit' cuando se te pida input para terminar.\n")

    try:
        player_proxy.initiate_chat(
            manager,
            message="*Entro corriendo al palacio* ¡Jarl Balgruuf, Farengar! ¡Tengo noticias urgentes, he visto un dragón negro destruyendo Helgen!"
        )
    except KeyboardInterrupt:
        print("\nCerrando consola de pruebas...")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    iniciar_consola_ag2()