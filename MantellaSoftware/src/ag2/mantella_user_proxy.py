
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
        # Solo se invoca con human_input_mode != "NEVER". En producción este proxy
        # siempre usa "NEVER" (la entrada del jugador llega por streaming_call), así
        # que este método no tiene ninguna fuente de transcripción conectada.
        import logging
        logging.error(
            "[User Proxy] get_human_input() invocado con human_input_mode != 'NEVER', "
            "pero no hay ninguna fuente de entrada humana conectada a MantellaUserProxy."
        )
        raise NotImplementedError(
            "MantellaUserProxy.get_human_input no está conectado a ninguna fuente de "
            "entrada (STT/texto). Usa human_input_mode='NEVER' o inyecta una fuente de "
            "transcripción por el constructor antes de usar este modo."
        )
