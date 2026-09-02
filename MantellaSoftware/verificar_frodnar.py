import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.my_custom_client import MyCustomClient
from src.config.definitions.llm_definitions import NarrationIndicatorsEnum


class MockConfig:
    def __init__(self):
        self.llm_api = "http://127.0.0.1:11434/v1"
        self.llm = "gemma4:e2b"
        self.llm_params = {"temperature": 0.8, "max_tokens": 800, "stop": ["<TERMINAR>"]}
        self.custom_token_count = 4096
        self.vision_enabled = False
        self.advanced_actions_enabled = False
        self.narration_indicators = NarrationIndicatorsEnum.PARANTHESES


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


PREGUNTAS = [
    "¿Quién es Son Goku?",
    "¿Qué son las Bolas de Dragón?",
    "¿Cuál es tu plato favorito para desayunar?",  # control, sin relación
]


async def main():
    config = MockConfig()
    cliente = MyCustomClient(
        config=config,
        secret_key_file="GPT_SECRET_KEY.txt",
        image_secret_key_file="IMAGE_SECRET_KEY.txt"
    )
    print(f"[+] Documentos en la colección: {cliente.coleccion_memoria.count()}")

    for pregunta in PREGUNTAS:
        historial = [MockMessage("system", "Eres Frodnar."), MockMessage("user", pregunta)]
        thread = MockMessageThread(historial)
        print(f"\n--- Frodnar | {pregunta} ---")
        respuesta = ""
        async for chunk_type, chunk_data in cliente.streaming_call(thread, is_multi_npc=False):
            if chunk_type == "content" and chunk_data:
                respuesta += chunk_data
        print(f"Frodnar: {respuesta.strip()}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
