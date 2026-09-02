import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ag2.ag2_client import AG2Client
from src.ag2.mantella_user_proxy import MantellaUserProxy
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

NPC_NOMBRE = "Frodnar_TestMemoria"


# --- Mocks mínimos que imitan la forma real que usa AG2Client/AgentFactory ---
class MockCharacter:
    def __init__(self, name, ref_id="npc_1"):
        self.name = name
        self.ref_id = ref_id
        self.is_player_character = False
        self.bio = "Un niño de Piedra del Río."


class MockCharacters:
    def __init__(self, characters):
        self._characters = characters
    def get_all_characters(self):
        return self._characters
    def get_player_character(self):
        return None


class MockConfig:
    def __init__(self):
        self.llm_api = "http://127.0.0.1:11434/v1"
        self.llm = "gemma4:e2b"
        self.llm_params = {"temperature": 0.7}


class MockContext:
    def __init__(self, characters):
        self.npcs_in_conversation = MockCharacters(characters)
        self.config = MockConfig()


class MockMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    def get_openai_message(self):
        return {"role": self.role, "content": self.content}
    def get_formatted_content(self):
        return self.content


class MockMessageThread:
    def __init__(self, msgs):
        self._msgs = msgs
    def get_openai_messages(self):
        return [m.get_openai_message() for m in self._msgs]
    def get_last_message(self):
        return self._msgs[-1] if self._msgs else None


async def simular_conversacion(turno_texto, nota):
    print(f"\n{'='*60}\n{nota}\n{'='*60}")

    # Igual que GestorMemoriaAG2() sin argumentos en AG2Conversation real
    gestor_memoria = GestorMemoriaAG2()
    personaje = MockCharacter(NPC_NOMBRE)
    context = MockContext([personaje])
    user_proxy = MantellaUserProxy(name="Jugador", human_input_mode="NEVER", max_consecutive_auto_reply=0)

    # Como en producción: un AG2Client nuevo por conversación
    ag2_client = AG2Client(context=context, llm_client=None, gestor_memoria=gestor_memoria, user_proxy=user_proxy)

    mensajes = [
        MockMsg("system", f"You are {NPC_NOMBRE}, and you live in Skyrim."),
        MockMsg("user", turno_texto),
    ]
    thread = MockMessageThread(mensajes)

    respuesta = ""
    async for chunk in ag2_client.streaming_call(thread, is_multi_npc=False):
        if chunk:
            respuesta = chunk
    print(f"NPC: {respuesta}")

    # Igual que AG2Conversation.end()
    ag2_client.guardar_memorias()
    print("[+] Memoria guardada.")


async def main():
    # Limpiamos cualquier memoria previa de pruebas anteriores para que el test sea reproducible
    gestor = GestorMemoriaAG2()
    ruta = gestor.obtener_ruta_archivo(NPC_NOMBRE)
    if os.path.exists(ruta):
        os.remove(ruta)

    await simular_conversacion(
        "Hola, me llamo Aela y quiero que sepas que mi color favorito es el verde.",
        "CONVERSACIÓN 1 (sin memoria previa)"
    )
    await simular_conversacion(
        "¿Recuerdas cómo me llamo y cuál es mi color favorito?",
        "CONVERSACIÓN 2 (nueva instancia de AG2Client, debe cargar la memoria de la conversación 1)"
    )


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
