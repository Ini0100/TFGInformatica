from typing import AsyncGenerator, Any
from src.llm.ai_client import AIClient
from src.llm.messages import Message
from src.llm.message_thread import message_thread
from src.llm.llm_model_list import LLMModelList
from autogen import AssistantAgent, UserProxyAgent
from src.ag2.agent_factory import AgentFactory
import logging
import re

class AG2Client(AIClient):

    def __init__(self, context, llm_client, gestor_memoria, user_proxy: UserProxyAgent):
        self.context = context
        self.llm_client = llm_client
        self.gestor_memoria = gestor_memoria
        self.user_proxy = user_proxy
        self.agents = {}
        self.character_names = {}
        self._encoding = None 

    def request_call(self, messages: Message | message_thread) -> str | None:
        return None

    def _extraer_nombre_npc_seguro(self, messages) -> str:
        """Extrae el nombre del NPC directamente del prompt del sistema como plan de respaldo."""
        try:
            if hasattr(messages, 'messages') and isinstance(messages.messages, list):
                for msg in messages.messages:
                    if getattr(msg, 'role', '') == 'system' or (hasattr(msg, 'get_openai_message') and msg.get_openai_message().get('role') == 'system'):
                        contenido = getattr(msg, 'content', '') if hasattr(msg, 'content') else msg.get_formatted_content()
                        # Busca "You are [Nombre]" o "Eres [Nombre]"
                        match = re.search(r'(?:You are|Eres)\s+([^,\.]+)', contenido, re.IGNORECASE)
                        if match:
                            return match.group(1).strip()
        except Exception as e:
            logging.error(f"AG2Client: Error extrayendo nombre seguro: {e}")
        return None

    async def streaming_call(self, messages: Message | message_thread, is_multi_npc: bool, tools: list[dict] | None = None) -> AsyncGenerator[str | None, None]:
        
        # --- INTENTO 1: Carga desde el contexto oficial ---
        try:
            for character in self.context.npcs_in_conversation.get_all_characters():
                if getattr(character, 'is_player', False) or getattr(character, 'is_player_character', False):
                    continue
                    
                if character.ref_id not in self.agents:
                    agent = AgentFactory.create_agent(
                        character=character, 
                        config=self.context.config,
                        llm_client=self.llm_client,                     
                        gestor_memoria=self.gestor_memoria         
                    )
                    if agent:
                        self.agents[character.ref_id] = agent
                        self.character_names[character.ref_id] = character.name
                        logging.info(f"AG2: Agente creado exitosamente para {character.name}")
        except Exception as e:
            logging.error(f"AG2: Fallo al intentar cargar agentes desde el contexto: {e}")

        # Reemplaza el retorno estricto por este bloque de rescate
        if not self.agents:
            logging.warning("AG2: Array de agentes vacío. Forzando carga desde los actores del contexto...")
            
            npc_target = None
            # Buscamos al NPC en la lista de actores del contexto
            if hasattr(self.context, 'actors'):
                for actor in self.context.actors:
                    if not getattr(actor, 'is_player', False):
                        npc_target = actor
                        break
            
            if npc_target:
                logging.info(f"AG2: Creando agente de emergencia para {npc_target.name}")
                # Llama a tu factoría para instanciar al agente en tiempo real
                agente_emergencia = AgentFactory.create_agent(
                    character=npc_target,
                    config=self.context.config,
                    llm_client=self.llm_client,
                    gestor_memoria=self.gestor_memoria # Si lo tienes disponible aquí
                )
                if agente_emergencia:
                    self.agents[npc_target.ref_id] = agente_emergencia
                else:
                    yield "System Error: No se pudo instanciar al agente."
                    return
            else:
                yield "No agents available." # Solo falla si realmente no hay ningún NPC cerca
                return


        # Seleccionar agente
        agent_ref_id = next(iter(self.agents)) 
        agent = self.agents[agent_ref_id]
        character_name = self.character_names.get(agent_ref_id, agent.name)

        last_message = messages.get_last_message()
        last_content = last_message.get_formatted_content() if last_message else ""
        
        try:
            logging.info(f"AG2: Solicitando respuesta a AutoGen para el agente {agent.name}")
            
            # Generar respuesta bloqueante
            reply = agent.generate_reply(messages=[{"role": "user", "content": last_content}])
            
            if reply and isinstance(reply, str):
                yield f"{character_name}: {reply}"
            elif isinstance(reply, dict) and "content" in reply:
                 yield f"{character_name}: {reply['content']}"
            else:
                logging.warning(f"AG2: Agent returned non-string reply: {reply}")

        except Exception as e:
            logging.error(f"AG2 Error Crítico durante la generación: {e}")
            yield "I am having trouble thinking right now."

    def get_count_tokens(self, messages: message_thread | list[Message] | Message | str) -> int:
        return 0

    def is_too_long(self, messages: message_thread | list[Message] | Message | str, token_limit_percent: float) -> bool:
        return False

    @staticmethod
    def get_model_list(service: str, secret_key_file: str, default_model: str = "gemini-flash-latest", is_vision: bool = False, is_tool_calling: bool = False) -> LLMModelList:
         opciones = ["gemini-flash-latest", "AG2 Agent System", default_model]
         return LLMModelList(opciones, default_model)
