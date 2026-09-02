from autogen import AssistantAgent
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.llm.client_base import ClientBase
import logging

class AgentFactory:
    """Factory for creating AG2 agents from Mantella characters."""

    @staticmethod
    def create_agent(character: Character, config: ConfigLoader, llm_client=None, gestor_memoria=None, system_message_override: str | None = None) -> AssistantAgent:
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

            # Usamos el prompt completo que ya construye Mantella (idioma, reglas de roleplay,
            # hora, clima, etc.) en vez de uno recortado, para que AG2 tenga el mismo contexto
            # que el cliente estandar. Si no esta disponible, caemos al prompt minimo anterior.
            if system_message_override:
                system_message = f"{system_message_override}\n\nAdditional Memory/Context:\n{contexto_rag}"
            else:
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
        # AG2 no reconoce "gemini" como api_type valido (solo "google", que exige el SDK nativo).
        # Como Mantella habla con Gemini via su endpoint compatible con OpenAI
        # (https://generativelanguage.googleapis.com/v1beta/openai), dejamos que AG2 use
        # su wrapper "openai" por defecto en vez de forzar un api_type inexistente.
        api_type = None

        if service == 'OpenAI':
            base_url = "https://api.openai.com/v1"
        elif service == 'OpenRouter':
            base_url = "https://openrouter.ai/api/v1"
        elif service == 'Kobold':
            base_url = "http://127.0.0.1:5001/v1"
        else:
            base_url = service

        # config.llm guarda el nombre "bonito" para mostrar en la UI (ej. "Gemini 1.5 Flash"),
        # no el ID real que espera la API (ej. "gemini-1.5-flash"). client_base.py ya corrige
        # esto para el cliente estandar; replicamos la misma correccion aqui para AG2.
        if ("googleapis" in base_url or "gemini" in base_url) and not model_name.lower().startswith("gemini-"):
            logging.info(f"[AG2 Config] Detectada URL de Gemini pero modelo invalido '{model_name}'. Forzando a 'gemini-flash-latest'.")
            model_name = "gemini-flash-latest"

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
            # config.llm_params puede ser None si el JSON de "Parameters" en la
            # config está mal formado (ver el mismo patrón defensivo en
            # ClientBase.max_tokens_param); sin esta guarda, un config.ini con
            # ese campo inválido tumbaría cualquier conversación con AG2.
            "temperature": config.llm_params.get("temperature", 0.7) if config.llm_params else 0.7,
            "cache_seed": 42,
        }

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c == '_')
