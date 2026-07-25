
import time
from autogen import UserProxyAgent
from typing import Optional

class MantellaUserProxy(UserProxyAgent):
    """
    Custom User Proxy Agent for Mantella.
    This acts as the bridge between the human player (via Mantella's STT/Input) and the AG2 agent ecosystem.
    """

    def __init__(
        self,
        name: str = "Player",
        system_message: Optional[str] = "You are the player interacting with the NPCs.",
        human_input_mode: str = "ALWAYS",
        max_consecutive_auto_reply: int = 0,
        **kwargs
    ):
        """
        Args:
            name (str): The name of the player agent.
            system_message (str): The role description for the player proxy.
            human_input_mode (str): Mode for human input. Default is ALWAYS (ask for input).
            max_consecutive_auto_reply (int): Max auto replies before asking for input.
            **kwargs: Additional args for UserProxyAgent.
        """
        # Remove code_execution_config from kwargs if present to avoid collision with our hardcoded value
        if 'code_execution_config' in kwargs:
            kwargs.pop('code_execution_config')

        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=False, # Disable code execution for now
            **kwargs
        )

    def get_human_input(self, prompt: str) -> str:
        # 1. Usamos logging en lugar de un print básico para que quede en el historial
        import logging
        logging.info(f"[User Proxy] AutoGen solicita input del jugador: {prompt}") 
        
        # 2. Implementar una espera no bloqueante (polling) en el hilo secundario
        while True:
            # Consultamos la cola de entrada de audio de Mantella
            # Nota: 'stt' hace referencia al módulo Speech-to-Text configurado en tu entorno
            player_input = stt.get_latest_transcription() 
            
            # Si Mantella detecta y transcribe la voz del jugador, rompemos el bucle y devolvemos el texto
            if player_input:
                logging.info(f"[User Proxy] Voz detectada y enviada a AG2: {player_input}")
                return player_input
                
            # 3. Pausa mínima para liberar recursos de la CPU y evitar bloqueos en el juego
            time.sleep(0.5)
