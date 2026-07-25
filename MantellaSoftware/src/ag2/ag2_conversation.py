from src.conversation.conversation import Conversation
from src.ag2.agent_factory import AgentFactory
from src.ag2.mantella_user_proxy import MantellaUserProxy
from src.ag2.ag2_client import AG2Client
from src.output_manager import ChatManager
import logging

class AG2Conversation(Conversation):
    """
    Clase adaptadora para integrar AG2 (AutoGen) en el flujo de Mantella.
    """
    gestor_memoria = None 

    def __init__(self, context_for_conversation, output_manager, rememberer, llm_client, stt, mic_input, mic_ptt):
        
        # 1. Proxy del usuario (jugador)
        player_char = context_for_conversation.npcs_in_conversation.get_player_character()
        player_name = player_char.name if player_char else "Player"
        
        user_proxy = MantellaUserProxy(
            name=player_name,
            human_input_mode="NEVER", 
            max_consecutive_auto_reply=0
        )

        # 2. Cliente AG2 (Ahora recibe el contexto completo, no el diccionario de agentes vacío)
        ag2_client = AG2Client(
            context=context_for_conversation, 
            llm_client=llm_client, 
            gestor_memoria=self.gestor_memoria,
            user_proxy=user_proxy
        )

        # 3. Gestor de chat puente
        ag2_chat_manager = ChatManager(context_for_conversation.config, output_manager.tts, ag2_client)

        # 4. Inicializamos la conversación base
        super().__init__(
            context_for_conversation, 
            ag2_chat_manager, 
            rememberer, 
            ag2_client, 
            stt, 
            mic_input, 
            mic_ptt
        )