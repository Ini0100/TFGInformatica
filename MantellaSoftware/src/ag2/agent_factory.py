from autogen import AssistantAgent
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.llm.client_base import ClientBase
import logging

class AgentFactory:
    """Factory for creating AG2 agents from Mantella characters."""

    @staticmethod
    def create_agent(character: Character, config: ConfigLoader, llm_client=None, gestor_memoria=None) -> AssistantAgent:
        """
        Creates an AG2 AssistantAgent representing the given Mantella character.
        """
        try:
            llm_config = AgentFactory._get_llm_config(config)
            
            # 1. Integramos la memoria RAG (ChromaDB) en el System Message
            contexto_rag = ""
            if gestor_memoria:
                try:
                    contexto_rag = gestor_memoria.obtener_contexto(character.name) 
                except Exception as e:
                    logging.warning(f"No se pudo cargar contexto RAG para {character.name}: {e}")

            # Inyectamos la memoria vectorial directamente en el cerebro del agente
            system_message = f"You are {character.name}. {character.bio}\n\nAdditional Memory/Context:\n{contexto_rag}"

            agent = AssistantAgent(
                name=AgentFactory._sanitize_name(character.name),
                system_message=system_message,
                llm_config=llm_config,
            )
            return agent
        except Exception as e:
            logging.error(f"AgentFactory: Error creating agent for {character.name}: {e}")
            return None

    @staticmethod
    def _get_llm_config(config: ConfigLoader) -> dict:
        """
        Constructs the LLM configuration for AG2 based on Mantella's config.
        """
        secret_key_file = 'GPT_SECRET_KEY.txt' 
        api_key = ClientBase._get_api_key(secret_key_file, show_error=False)
        
        try:
             service = config.llm_api 
             model_name = config.llm 
        except AttributeError:
             logging.warning("AgentFactory: Could not find llm_api or llm in config, checks definitions.")
             service = config.definitions.get_value("llm_api")
             model_name = config.definitions.get_value("model")

        base_url = None
        
        # 2. Arreglamos la compatibilidad de AG2 con Gemini
        api_type = None 
        if "gemini" in model_name.lower():
            api_type = "gemini"  # Le dice a AutoGen que use el wrapper de Google

        if service == 'OpenAI':
            base_url = "https://api.openai.com/v1"
        elif service == 'OpenRouter':
            base_url = "https://openrouter.ai/api/v1"
        elif service == 'Kobold':
            base_url = "http://127.0.0.1:5001/v1"
        else:
            base_url = service 

        # 3. Añadimos el api_type a la configuración si existe
        config_dict = {
            "model": model_name,
            "api_key": api_key if api_key else "NULL",
            "base_url": base_url,
        }
        
        if api_type:
            config_dict["api_type"] = api_type

        config_list = [config_dict]

        logging.info(f"[AG2 Config] Inicializando configuración del LLM - Modelo asignado: {model_name} | API Type: {api_type}")

        return {
            "config_list": config_list,
            "temperature": config.llm_params.get("temperature", 0.7), 
            "cache_seed": 42, 
        }

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c == '_')
