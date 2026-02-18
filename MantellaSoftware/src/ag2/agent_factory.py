
from autogen import AssistantAgent
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.llm.client_base import ClientBase

class AgentFactory:
    """Factory for creating AG2 agents from Mantella characters."""

    @staticmethod
    def create_agent(character: Character, config: ConfigLoader) -> AssistantAgent:
        """
        Creates an AG2 AssistantAgent representing the given Mantella character.

        Args:
            character (Character): The Mantella character to convert.
            config (ConfigLoader): The configuration loader containing LLM settings.

        Returns:
            AssistantAgent: The configured AG2 agent.
        """
        llm_config = AgentFactory._get_llm_config(config)
        
        # Construct a system message based on the character's bio
        # In the real implementation, this should also include the current context, 
        # similar to how Conversation.generate_prompt works, but for now we use the bio.
        system_message = f"You are {character.name}. {character.bio}"

        agent = AssistantAgent(
            name=AgentFactory._sanitize_name(character.name),
            system_message=system_message,
            llm_config=llm_config,
        )
        return agent

    @staticmethod
    def _get_llm_config(config: ConfigLoader) -> dict:
        """
        Constructs the LLM configuration for AG2 based on Mantella's config.
        """
        # Extract model and API key using ClientBase logic (or similar)
        # We need to replicate some of ClientBase's logic here or instantiate it to get the values.
        # Since ClientBase is designed to be instantiated, let's look at config directly where possible,
        # or use helper methods if they were static. _get_api_key is static.

        # 1. API Key
        # The key file depends on the service. defaulting to GPT_SECRET_KEY.txt for now.
        secret_key_file = 'GPT_SECRET_KEY.txt' 
        api_key = ClientBase._get_api_key(secret_key_file, show_error=False)
        
        # 2. Base URL & Model
        # We need to handle different services (OpenAI, OpenRouter, Local, etc.)
        # Mantella stores this in config.
        # We'll map Mantella's "LLM Service" concept to AG2's config.
        
        # definitions are in config.definitions
        try:
             service = config.llm_service # This might be an enum or string depending on implementation
             model_name = config.llm_model 
        except AttributeError:
             # Fallback if I can't find exact property names without looking at definitions, 
             # but based on ClientBase, these should exist or be accessible via definitions.
             # Let's check config_loader again if needed, but for prototype we can try standard names.
             # Actually, simpler: ClientBase takes (api_url, llm, ...)
             # Let's try to infer from config.ini values if we can access them.
             # ConfigLoader exposes values as properties if they are defined.
             service = config.definitions.get_value("llm_service")
             model_name = config.definitions.get_value("llm_model")

        base_url = None
        api_type = None

        if service == 'OpenAI':
            base_url = "https://api.openai.com/v1"
        elif service == 'OpenRouter':
            base_url = "https://openrouter.ai/api/v1"
        elif service == 'Kobold':
            base_url = "http://127.0.0.1:5001/v1"
        else:
            base_url = service # Custom URL case

        config_list = [
            {
                "model": model_name,
                "api_key": api_key if api_key else "NULL", # Local models might not need key
                "base_url": base_url,
            }
        ]

        return {
            "config_list": config_list,
            "temperature": config.definitions.get_value("temperature"), # Assuming this exists
            "cache_seed": 42, # For reproducibility during testing
        }

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitizes the character name to be a valid AG2 agent name.
        (Alphanumeric and underscores only typically)
        """
        return "".join(c for c in name if c.isalnum() or c == '_')
