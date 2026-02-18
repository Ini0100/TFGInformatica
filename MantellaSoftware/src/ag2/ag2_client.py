
from typing import AsyncGenerator, Any
from src.llm.ai_client import AIClient
from src.llm.messages import Message
from src.llm.message_thread import message_thread
from src.llm.llm_model_list import LLMModelList
from autogen import AssistantAgent, UserProxyAgent
import logging

class AG2Client(AIClient):
    """
    Adapter class that exposes AG2 agents as a Mantella AIClient.
    This allows ChatManager to 'stream' responses from AG2 agents without knowing they are agents.
    """

    def __init__(self, agents: dict[str, AssistantAgent], character_names: dict[str, str], user_proxy: UserProxyAgent):
        """
        Args:
            agents: Dict of ref_id -> AssistantAgent
            character_names: Dict of ref_id -> Character Name (as Mantella expects)
            user_proxy: The user proxy agent
        """
        self.agents = agents
        self.character_names = character_names
        self.user_proxy = user_proxy
        # Token counting fallback
        self._encoding = None 
        # We might need the base client's encoding logic, but simplest is to just use len or approx 
        # or instantiate a dummy TikToken encoder if strictly needed.

    def request_call(self, messages: Message | message_thread) -> str | None:
        """Synchronous call - not typically used in the main loop but implemented for safety."""
        # For now, just call generate_reply on the first available agent
        # logic should match streaming_call but blocking
        # This might be used for 'summaries' or other non-chat calls. 
        # If so, we might want to delegate to the original LLMClient? 
        # For now, return None or basic string to indicate AG2 is active.
        return None

    async def streaming_call(self, messages: Message | message_thread, is_multi_npc: bool) -> AsyncGenerator[str | None, None]:
        """
        Triggers an AG2 agent reply and yields it as a stream.
        """
        # 1. Determine which agent should speak.
        # In a real implementation, we'd use a GroupChatManager or analyze 'messages'.
        # For this prototype: Pick the first NPC agent (Action selection logic needed later).
        if not self.agents:
            yield "No agents available."
            return

        # Simple round-robin or just pick the first one for 1-on-1
        agent_ref_id = next(iter(self.agents)) 
        agent = self.agents[agent_ref_id]
        character_name = self.character_names.get(agent_ref_id, agent.name)

        # 2. Prepare context for the agent
        # AG2 agents manage their own history. However, we need to ensure the *current* user input 
        # (from 'messages') is passed to them if they haven't seen it.
        # Mantella's 'messages' contains the full history including the latest user message.
        
        last_message = messages.get_last_message()
        last_content = last_message.get_content() if last_message else ""
        
        # 3. Generate Reply
        # We use a blocking call to generate_reply (AG2 is mostly sync, but has async support).
        # We wrap it in the async generator.
        
        try:
            # Note: generate_reply might be blocking. In production, run in executor if needed.
            logging.info(f"AG2: Requesting reply from {agent.name}")
            
            # We construct a message list for the generation
            # If we trust the agent's internal memory, we just send the new user message.
            # If we need to stateless-ly force it, we pass full history.
            # Let's try passing the last user message.
            
            reply = agent.generate_reply(messages=[{"role": "user", "content": last_content}])
            
            if reply and isinstance(reply, str):
                # Yield the full reply prefixed with the character name so ChangeCharacterParser picks it up
                yield f"{character_name}: {reply}"
            elif isinstance(reply, dict) and "content" in reply:
                 yield f"{character_name}: {reply['content']}"
            else:
                logging.warning(f"AG2: Agent returned non-string reply: {reply}")

        except Exception as e:
            logging.error(f"AG2 Error: {e}")
            yield "I am having trouble thinking right now."

    def get_count_tokens(self, messages: message_thread | list[Message] | Message | str) -> int:
        # Dummy implementation
        return 0

    def is_too_long(self, messages: message_thread | list[Message] | Message | str, token_limit_percent: float) -> bool:
        # Always return False for now, assuming AG2 manages its own context window
        return False

    @staticmethod
    def get_model_list(service: str, secret_key_file: str, default_model: str = "google/gemma-3-27b-it:free", is_vision: bool = False, is_tool_calling: bool = False) -> LLMModelList:
         return LLMModelList([], "AG2 Agent System")
