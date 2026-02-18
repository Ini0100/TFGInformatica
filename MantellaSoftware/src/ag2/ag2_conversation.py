
from src.conversation.conversation import Conversation
from src.ag2.agent_factory import AgentFactory
from src.ag2.mantella_user_proxy import MantellaUserProxy
from src.ag2.ag2_client import AG2Client
from src.output_manager import ChatManager
import logging

class AG2Conversation(Conversation):
    """
    Adapter class to integreate AG2 (AutoGen) into Mantella's conversation flow.
    Inherits from Conversation to ensure compatibility with GameStateManager.
    """

    def __init__(self, context_for_conversation, output_manager, rememberer, llm_client, stt, mic_input, mic_ptt):
        
        # 1. Initialize AG2 components
        agents = {}
        character_names = {}
        
        # Create User Proxy
        player_char = context_for_conversation.npcs_in_conversation.get_player_character()
        player_name = player_char.name if player_char else "Player"
        
        # Note: MantellaUserProxy instance might need to be stored if we use it for input later.
        # But AG2Client holds it for now.
        user_proxy = MantellaUserProxy(
            name=player_name,
            human_input_mode="NEVER", 
            max_consecutive_auto_reply=0
        )

        # Create Agents for NPCs
        for character in context_for_conversation.npcs_in_conversation.get_all_characters():
            if character.is_player_character:
                # Add player to character_names map as well, just in case
                character_names[character.ref_id] = character.name
                continue
            
            # Use Factory to create agent
            # We pass the context's config.
            agent = AgentFactory.create_agent(character, context_for_conversation.config)
            
            agents[character.ref_id] = agent
            character_names[character.ref_id] = character.name
            
            logging.info(f"Initialized AG2 agent for {character.name}")

        # 2. Create AG2Adapter Client
        ag2_client = AG2Client(agents, character_names, user_proxy)

        # 3. Create a NEW ChatManager that uses the AG2Client
        # We reuse the TTS engine from the passed output_manager
        ag2_chat_manager = ChatManager(context_for_conversation.config, output_manager.tts, ag2_client)

        # 4. Initialize the base Conversation with the new components
        # We pass ag2_client as the llm_client, and ag2_chat_manager as output_manager
        super().__init__(
            context_for_conversation, 
            ag2_chat_manager, 
            rememberer, 
            ag2_client, 
            stt, 
            mic_input, 
            mic_ptt
        )
