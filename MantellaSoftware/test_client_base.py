import asyncio
import sys
import os
from unittest.mock import MagicMock

# Como estamos en Linux y Mantella tiene dependencias de Windows en utils.py, 
# mockeamos 'winsound' y 'winreg' para que no rompa al importar ClientBase
sys.modules['winsound'] = MagicMock()
sys.modules['winreg'] = MagicMock()
sys.modules['src.actions.function_manager'] = MagicMock()

# Añadimos la raíz al path para que las importaciones de src.* funcionen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.llm.client_base import ClientBase

# Creamos clases Mock para evitar cargar los objetos complejos del sistema de Mantella
class MockMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def get_openai_message(self) -> dict:
        return {"role": self.role, "content": self.content}

    def get_formatted_content(self) -> str:
        return self.content

class MockMessageThread:
    def __init__(self, messages):
        self.messages = messages
    
    def get_openai_messages(self):
        return [m.get_openai_message() for m in self.messages]
        
    def get_last_message(self):
        return self.messages[-1]

async def run_test():
    # 1. Parámetros de inicialización base
    api_url = "https://generativelanguage.googleapis.com/v1beta/openai" 
    llm_model = "gemini-flash-latest"
    llm_params = {"temperature": 0.7, "max_tokens": 150}
    custom_token_count = 4096
    secret_key_files = ["GPT_SECRET_KEY.txt"]

    print("Inicializando ClientBase en un entorno aislado...")
    client = ClientBase(
        api_url=api_url,
        llm=llm_model,
        llm_params=llm_params,
        custom_token_count=custom_token_count,
        secret_key_files=secret_key_files
    )

    print(f"Modelo configurado: {client.model_name}")
    api_k = client.api_key
    print(f"La API key encontrada comienza con: {api_k[:5] if api_k else 'Ninguna'}...")
    if not api_k or api_k == "abc123":
        print("\nADVERTENCIA: No se encontró una clave GPT_SECRET_KEY.txt válida.")
    
    # 2. Mensajes para la prueba
    thread = MockMessageThread([
        MockMessage("system", "Eres un asistente amable."),
        MockMessage("user", "¿De qué color es el cielo? Responde en 2 palabras.")
    ])

    # --------- PRUEBA SÍNCRONA ---------
    print(f"\n[!] Lanzando petición síncrona...")
    try:
        respuesta_sincrona = client.request_call(thread)
        print("\n--- Respuesta Síncrona ---")
        print(respuesta_sincrona)
        print("--------------------------\n")
    except Exception as e:
        print(f"Error Síncrono: {e}")

    # --------- PRUEBA ASÍNCRONA (Streaming) ---------
    print("\n[!] Lanzando petición asíncrona (streaming)...")
    try:
        print("\n--- Respuesta Streaming ---")
        async for type_chunk, chunk in client.streaming_call(thread, is_multi_npc=False):
            if type_chunk == "content" and chunk is not None:
                print(chunk, end="", flush=True)
        print("\n---------------------------\n")
    except Exception as e:
        print(f"Error Asíncrono: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
