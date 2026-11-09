from typing import AsyncGenerator, Any
from src.llm.ai_client import AIClient
from src.llm.messages import Message
from src.llm.message_thread import message_thread
from src.llm.llm_model_list import LLMModelList
from autogen import AssistantAgent, UserProxyAgent
from src.ag2.agent_factory import AgentFactory
import logging
import re
import requests

class AG2Client(AIClient):

    EMOCION_PRODUCCION = "neutral"

    def __init__(self, context, llm_client, gestor_memoria, user_proxy: UserProxyAgent):
        self.context = context
        self.llm_client = llm_client
        self.gestor_memoria = gestor_memoria
        self.user_proxy = user_proxy
        self.agents = {}
        self.character_names = {}
        self.historial_previo: dict[Any, list[dict]] = {}
        self.last_messages: message_thread | None = None
        self._encoding = None

    def request_call(self, messages: Message | message_thread) -> str | None:
        return self.llm_client.request_call(messages)

    def _extraer_nombre_npc_seguro(self, messages) -> str:
        """Extrae el nombre del NPC directamente del prompt del sistema como plan de respaldo."""
        try:
            if hasattr(messages, 'messages') and isinstance(messages.messages, list):
                for msg in messages.messages:
                    if getattr(msg, 'role', '') == 'system' or (hasattr(msg, 'get_openai_message') and msg.get_openai_message().get('role') == 'system'):
                        contenido = getattr(msg, 'content', '') if hasattr(msg, 'content') else msg.get_formatted_content()
                        match = re.search(r'(?:You are|Eres)\s+([^,\.]+)', contenido, re.IGNORECASE)
                        if match:
                            return match.group(1).strip()
        except Exception as e:
            logging.error(f"AG2Client: Error extrayendo nombre seguro: {e}")
        return None

    def _extraer_system_prompt_real(self, messages) -> str | None:
        """Extrae el system prompt completo (idioma, rol, hora, clima, reglas...) que Mantella
        ya construye para el flujo estandar, para que AG2 lo use en vez de un prompt recortado."""
        try:
            if hasattr(messages, 'get_openai_messages'):
                for m in messages.get_openai_messages():
                    if m.get('role') == 'system':
                        return m.get('content')
        except Exception as e:
            logging.error(f"AG2Client: Error extrayendo el system prompt real: {e}")
        return None

    def _extraer_texto_memoria(self, historial: list[dict]) -> str:
        """Extrae el texto plano del resumen guardado por GestorMemoriaAG2, para
        inyectarlo en el system_message del agente (ver AgentFactory.create_agent)."""
        return " ".join(m.get("content", "") for m in historial if m.get("content"))

    async def streaming_call(self, messages: Message | message_thread, is_multi_npc: bool, tools: list[dict] | None = None) -> AsyncGenerator[str | None, None]:

        self.last_messages = messages

        system_prompt_real = self._extraer_system_prompt_real(messages)

        try:
            for character in self.context.npcs_in_conversation.get_all_characters():
                if getattr(character, 'is_player', False) or getattr(character, 'is_player_character', False):
                    continue

                if character.ref_id not in self.agents:
                    historial_cargado = self.gestor_memoria.cargar_conversacion(
                        character.name, self.EMOCION_PRODUCCION
                    ) if self.gestor_memoria else []
                    agent = AgentFactory.create_agent(
                        character=character,
                        config=self.context.config,
                        llm_client=self.llm_client,
                        gestor_memoria=self.gestor_memoria,
                        system_message_override=system_prompt_real,
                        memoria_previa=self._extraer_texto_memoria(historial_cargado)
                    )
                    if agent:
                        self.agents[character.ref_id] = agent
                        self.character_names[character.ref_id] = character.name
                        self.historial_previo[character.ref_id] = historial_cargado
                        logging.info(f"AG2: Agente creado exitosamente para {character.name}")
        except Exception as e:
            logging.error(f"AG2: Fallo al intentar cargar agentes desde el contexto: {e}")

        if not self.agents:
            logging.warning("AG2: Array de agentes vacío. Forzando carga desde los actores del contexto...")

            npc_target = None
            if hasattr(self.context, 'actors'):
                for actor in self.context.actors:
                    if not getattr(actor, 'is_player', False):
                        npc_target = actor
                        break

            if npc_target:
                logging.info(f"AG2: Creando agente de emergencia para {npc_target.name}")
                historial_cargado_emergencia = self.gestor_memoria.cargar_conversacion(
                    npc_target.name, self.EMOCION_PRODUCCION
                ) if self.gestor_memoria else []
                agente_emergencia = AgentFactory.create_agent(
                    character=npc_target,
                    config=self.context.config,
                    llm_client=self.llm_client,
                    gestor_memoria=self.gestor_memoria,
                    system_message_override=system_prompt_real,
                    memoria_previa=self._extraer_texto_memoria(historial_cargado_emergencia)
                )
                if agente_emergencia:
                    self.agents[npc_target.ref_id] = agente_emergencia
                    self.character_names[npc_target.ref_id] = npc_target.name
                    self.historial_previo[npc_target.ref_id] = historial_cargado_emergencia
                else:
                    yield "System Error: No se pudo instanciar al agente."
                    return
            else:
                yield "No agents available."
                return


        agent_ref_id = next(iter(self.agents))
        agent = self.agents[agent_ref_id]
        character_name = self.character_names.get(agent_ref_id, agent.name)

        openai_messages = messages.get_openai_messages()
        conversation_messages = [m for m in openai_messages if m.get("role") != "system"]
        if not conversation_messages:
            last_message = messages.get_last_message()
            last_content = last_message.get_formatted_content() if last_message else ""
            conversation_messages = [{"role": "user", "content": last_content}]

        try:
            logging.info(f"AG2: Solicitando respuesta a AutoGen para el agente {agent.name}")

            reply = await agent.a_generate_reply(messages=conversation_messages)

            if reply and isinstance(reply, str):
                yield f"{character_name}: {reply}"
            elif isinstance(reply, dict) and "content" in reply:
                 yield f"{character_name}: {reply['content']}"
            else:
                logging.warning(f"AG2: Agent returned non-string reply: {reply}")

        except Exception as e:
            logging.error(f"AG2 Error Crítico durante la generación: {e}")
            yield "I am having trouble thinking right now."

    RESUMEN_MAX_TOKENS = 700

    def _generar_resumen(self, historial: list[dict], npc_name: str) -> str:
        """Condensa el historial de conversación en 2-3 frases mediante una llamada al LLM."""
        if not historial:
            return ""
        texto_conversacion = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in historial if m.get("content"))
        if not texto_conversacion:
            return ""
        prompt_resumen = (
            f"Resume en 2-3 frases, en español y en tercera persona, los datos clave que "
            f"{npc_name} debería recordar de la siguiente conversación con el jugador "
            f"(nombres, hechos y temas mencionados). Sé breve y conciso.\n\n{texto_conversacion}"
        )
        try:
            base_url = getattr(self.context.config, "llm_api", "http://127.0.0.1:11434/v1")
            model_name = getattr(self.context.config, "llm", "gemma4:e2b")
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_resumen}],
                    "max_tokens": self.RESUMEN_MAX_TOKENS,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            resumen = resp.json()["choices"][0]["message"]["content"]
            return resumen.strip() if resumen else ""
        except Exception as e:
            logging.error(f"AG2: Error generando resumen de memoria para {npc_name}: {e}")
            return ""

    def guardar_memorias(self):
        """Persiste en disco (vía GestorMemoriaAG2) un resumen de la conversación
        mantenida con cada NPC que participó, en vez del historial completo.
        Llamado desde AG2Conversation.end() al finalizar la conversación."""
        if not self.gestor_memoria or not self.last_messages or not self.character_names:
            return

        historial_actual = [m for m in self.last_messages.get_openai_messages() if m.get("role") != "system"]
        if not historial_actual:
            return

        for ref_id, npc_name in self.character_names.items():
            try:
                historial_completo = self.historial_previo.get(ref_id, []) + historial_actual
                resumen = self._generar_resumen(historial_completo, npc_name)
                if resumen:
                    historial_a_guardar = [{
                        "role": "user",
                        "content": resumen,
                    }]
                    logging.info(f"AG2: Memoria de {npc_name} resumida y guardada ({len(resumen)} caracteres).")
                else:
                    historial_a_guardar = []
                    logging.warning(f"AG2: No se pudo generar un resumen para {npc_name}; no se guarda memoria de esta conversación.")
                self.gestor_memoria.guardar_conversacion(npc_name, historial_a_guardar, self.EMOCION_PRODUCCION)
            except Exception as e:
                logging.error(f"AG2: Error guardando memoria de {npc_name}: {e}")

    def get_count_tokens(self, messages: message_thread | list[Message] | Message | str) -> int:
        return self.llm_client.get_count_tokens(messages)

    def is_too_long(self, messages: message_thread | list[Message] | Message | str, token_limit_percent: float) -> bool:
        return self.llm_client.is_too_long(messages, token_limit_percent)

    @staticmethod
    def get_model_list(service: str, secret_key_file: str, default_model: str = "gemma4:e2b", is_vision: bool = False, is_tool_calling: bool = False) -> LLMModelList:
         opciones = ["gemma4:e2b", "AG2 Agent System", default_model]
         return LLMModelList(opciones, default_model)
