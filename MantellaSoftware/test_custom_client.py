import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IMPORTAMOS NUESTRO CLIENTE, NO EL CLIENTE BASE
from src.llm.my_custom_client import MyCustomClient

# Mocks para simular el entorno de Mantella sin abrir el juego
class MockConfig:
    def __init__(self):
        self.llm_api = "https://generativelanguage.googleapis.com/v1beta/openai"
        self.llm = "gemini-flash-latest"
        self.llm_params = {"temperature": 0.8, "max_tokens": 800}
        self.custom_token_count = 4096
        self.vision_enabled = False
        self.advanced_actions_enabled = False

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
    def add_message(self, message):
        self.messages.append(message)
    def add_system_message(self, text):
        self.messages.append(MockMessage("system", text))

async def chat_interactivo():
    print("\n[+] Iniciando MyCustomClient...")
    config = MockConfig()
    
    # Instanciamos nuestro cliente con memoria
    cliente = MyCustomClient(
        config=config,
        secret_key_file="GPT_SECRET_KEY.txt",
        image_secret_key_file="IMAGE_SECRET_KEY.txt"
    )

    historial_base = [MockMessage("system", "Eres Nazeem Fuego Secreto.")]

    print("\n=== CONSOLA DE PRUEBAS MANTELLA CON MEMORIA ===")
    print("Escribe 'salir' para terminar.\n")

    while True:
        texto_usuario = input("\nTú: ")
        if texto_usuario.lower() in ['salir', 'exit', 'quit']:
            break
        if not texto_usuario.strip():
            continue

        historial_turno = historial_base.copy()
        historial_turno.append(MockMessage("user", texto_usuario))
        thread = MockMessageThread(historial_turno)

        print("Farengar: ", end="", flush=True)
        
        try:
            # Aquí ocurre la magia: El cliente busca en memoria, inyecta y responde
            async for chunk_type, chunk_data in cliente.streaming_call(thread, is_multi_npc=False):
                if chunk_type == "content" and chunk_data:
                    print(chunk_data, end="", flush=True)
            print() 
            
        except Exception as e:
            print(f"\n[Error en el bucle principal]: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(chat_interactivo())
 
 
