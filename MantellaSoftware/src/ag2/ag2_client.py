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

    # Emoción usada para guardar_conversacion/cargar_conversacion. GestorMemoriaAG2
    # purga la memoria si la emoción guardada no coincide con la actual (pensado
    # para la capa emocional de Ekman de los scripts de prueba). El flujo real
    # del juego no tiene ese concepto de emoción por NPC, así que se usa un
    # valor fijo: con eso, guardar/cargar nunca provoca una purga y el NPC
    # simplemente recuerda lo hablado en la última conversación.
    EMOCION_PRODUCCION = "neutral"

    def __init__(self, context, llm_client, gestor_memoria, user_proxy: UserProxyAgent):
        self.context = context
        self.llm_client = llm_client
        self.gestor_memoria = gestor_memoria
        self.user_proxy = user_proxy
        self.agents = {}
        self.character_names = {}
        # Historial de conversaciones ANTERIORES con cada NPC, cargado de disco
        # al crear su agente. Se antepone al historial de la conversación en
        # curso en cada llamada a generate_reply (ver streaming_call).
        self.historial_previo: dict[Any, list[dict]] = {}
        self.last_messages: message_thread | None = None
        self._encoding = None

    def request_call(self, messages: Message | message_thread) -> str | None:
        # Usado por src/remember/summaries.py para generar el resumen de fin de
        # conversación ({conversation_summary} en el prompt). Antes devolvía
        # siempre None, lo que hacía fallar SIEMPRE ese resumen (en silencio,
        # solo un error en el log) para cualquier NPC llevado por AG2: nunca
        # se guardaba memoria a largo plazo estándar de Mantella para ellos.
        # Delegamos en el cliente LLM real (el mismo usado para RAG/HyDE), que
        # no depende de la maquinaria de agentes de AutoGen para esta llamada.
        return self.llm_client.request_call(messages)

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

    async def streaming_call(self, messages: Message | message_thread, is_multi_npc: bool, tools: list[dict] | None = None) -> AsyncGenerator[str | None, None]:

        # Guardado para poder persistir esta conversación al terminar (ver
        # guardar_memorias, llamado desde AG2Conversation.end()).
        self.last_messages = messages

        system_prompt_real = self._extraer_system_prompt_real(messages)

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
                        gestor_memoria=self.gestor_memoria,
                        system_message_override=system_prompt_real
                    )
                    if agent:
                        self.agents[character.ref_id] = agent
                        self.character_names[character.ref_id] = character.name
                        self.historial_previo[character.ref_id] = self.gestor_memoria.cargar_conversacion(
                            character.name, self.EMOCION_PRODUCCION
                        ) if self.gestor_memoria else []
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
                    gestor_memoria=self.gestor_memoria, # Si lo tienes disponible aquí
                    system_message_override=system_prompt_real
                )
                if agente_emergencia:
                    self.agents[npc_target.ref_id] = agente_emergencia
                    self.character_names[npc_target.ref_id] = npc_target.name
                    self.historial_previo[npc_target.ref_id] = self.gestor_memoria.cargar_conversacion(
                        npc_target.name, self.EMOCION_PRODUCCION
                    ) if self.gestor_memoria else []
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

        # IMPORTANTE: se pasa el hilo de conversación COMPLETO (sin el mensaje
        # de sistema, que AutoGen ya antepone solo desde agent.system_message),
        # no solo el último mensaje del jugador. Antes se llamaba a
        # generate_reply con una lista de un único mensaje en cada turno, lo
        # que hacía que el NPC "olvidara" todo lo dicho en turnos anteriores
        # DENTRO de la misma conversación (cada respuesta se generaba como si
        # fuera la primera línea del diálogo).
        openai_messages = messages.get_openai_messages()
        conversation_messages = [m for m in openai_messages if m.get("role") != "system"]
        if not conversation_messages:
            last_message = messages.get_last_message()
            last_content = last_message.get_formatted_content() if last_message else ""
            conversation_messages = [{"role": "user", "content": last_content}]

        # Antepone lo hablado en la ÚLTIMA conversación con este NPC (cargado de
        # disco al crear el agente), para que recuerde entre visitas separadas,
        # no solo dentro de la conversación actual.
        historial_previo = self.historial_previo.get(agent_ref_id, [])
        if historial_previo:
            conversation_messages = historial_previo + conversation_messages

        try:
            logging.info(f"AG2: Solicitando respuesta a AutoGen para el agente {agent.name}")

            # a_generate_reply (no generate_reply) porque streaming_call es async:
            # generate_reply es una llamada síncrona y bloqueante, y al no hacer
            # nunca un await dentro de esta función, el único yield ocurre al
            # terminar toda la generación. Esto rompía dos cosas en el juego real:
            # el chequeo de interrupción por micrófono en process_response()
            # (output_manager.py) solo se evalúa entre yields, así que con uno
            # solo al final el jugador no podía interrumpir a un NPC de AG2
            # mientras "pensaba"; y no había audio/subtítulos progresivos, solo
            # silencio hasta que aparecía la respuesta completa de golpe.
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

    def guardar_memorias(self):
        """Persiste en disco (vía GestorMemoriaAG2) la conversación mantenida con
        cada NPC que participó, para que la recuerden en la próxima conversación.
        Llamado desde AG2Conversation.end() al finalizar la conversación."""
        if not self.gestor_memoria or not self.last_messages or not self.character_names:
            return

        historial_actual = [m for m in self.last_messages.get_openai_messages() if m.get("role") != "system"]
        if not historial_actual:
            return

        for ref_id, npc_name in self.character_names.items():
            try:
                historial_completo = self.historial_previo.get(ref_id, []) + historial_actual
                self.gestor_memoria.guardar_conversacion(npc_name, historial_completo, self.EMOCION_PRODUCCION)
                logging.info(f"AG2: Memoria de {npc_name} guardada ({len(historial_completo)} mensajes).")
            except Exception as e:
                logging.error(f"AG2: Error guardando memoria de {npc_name}: {e}")

    def get_count_tokens(self, messages: message_thread | list[Message] | Message | str) -> int:
        # Delegado en el cliente LLM real: devolver 0 siempre desactivaba el
        # mecanismo estándar de Mantella que resume/recorta la conversación
        # cuando se acerca al límite de contexto del modelo (is_too_long nunca
        # se disparaba), lo que en una conversación larga podría acabar
        # desbordando el límite de tokens de gemma4:e2b sin previo aviso.
        return self.llm_client.get_count_tokens(messages)

    def is_too_long(self, messages: message_thread | list[Message] | Message | str, token_limit_percent: float) -> bool:
        return self.llm_client.is_too_long(messages, token_limit_percent)

    @staticmethod
    def get_model_list(service: str, secret_key_file: str, default_model: str = "gemma4:e2b", is_vision: bool = False, is_tool_calling: bool = False) -> LLMModelList:
         opciones = ["gemma4:e2b", "AG2 Agent System", default_model]
         return LLMModelList(opciones, default_model)
