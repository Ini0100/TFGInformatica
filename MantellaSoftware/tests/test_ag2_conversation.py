
import unittest
from unittest.mock import MagicMock
from src.ag2.agent_factory import AgentFactory
from src.ag2.mantella_user_proxy import MantellaUserProxy
from src.character_manager import Character
from src.config.config_loader import ConfigLoader

class TestAG2Conversation(unittest.TestCase):
    def setUp(self):
        # Mock ConfigLoader
        self.mock_config = MagicMock(spec=ConfigLoader)
        self.mock_config.definitions = MagicMock()
        # Mock specific config values needed by AgentFactory
        self.mock_config.definitions.get_value.side_effect = lambda key: {
            "llm_service": "OpenAI",
            "llm_model": "gpt-3.5-turbo",
            "temperature": 0.7
        }.get(key)
        self.mock_config.llm_service = "OpenAI"
        self.mock_config.llm_model = "gpt-3.5-turbo"

        # Mock Character (NPC)
        self.mock_npc = MagicMock(spec=Character)
        self.mock_npc.name = "Lydia"
        self.mock_npc.bio = "I am sworn to carry your burdens."
        
        # Mock Character (Player) - though not strictly needed for AgentFactory unless we make a player agent too
        self.mock_player = MagicMock(spec=Character)
        self.mock_player.name = "Dragonborn"

    def test_agent_creation(self):
        """Test that AgentFactory creates an agent with correct attributes."""
        agent = AgentFactory.create_agent(self.mock_npc, self.mock_config)
        
        self.assertEqual(agent.name, "Lydia")
        self.assertIn("Lydia", agent.system_message)
        self.assertIn("carry your burdens", agent.system_message)
        self.assertIsNotNone(agent.llm_config)
        # Verify llm_config structure
        self.assertIn("config_list", agent.llm_config)
        self.assertEqual(agent.llm_config["config_list"][0]["model"], "gpt-3.5-turbo")

    def test_conversation_flow(self):
        """Test a basic conversation flow between UserProxy and NPC Agent."""
        npc_agent = AgentFactory.create_agent(self.mock_npc, self.mock_config)
        
        # Create User Proxy
        user_proxy = MantellaUserProxy(
            name="Player",
            human_input_mode="NEVER", # Automated test
            max_consecutive_auto_reply=1,
            code_execution_config=False
        )
        
        # We need to mock the actual LLM call to avoid hitting API
        # AG2 agents use .generate_reply() or .receive() internally.
        # We can mock the openai connection in the agent's client, or simpler:
        # Mock the `clients` attribute of the agent if possible, or use `register_reply` to inject a mock response.
        
        # Inject a mock reply for the NPC agent
        npc_agent.register_reply(
            [user_proxy, None],
            lambda *args, **kwargs: (True, "I am sworn to carry your burdens, my Thane."), 
            position=0
        )
        
        # Initiate chat
        user_proxy.initiate_chat(
            npc_agent,
            message="Hello, Lydia."
        )
        
        # Verify conversation history
        # Agent saves history in .chat_messages[recipient_agent]
        history = user_proxy.chat_messages[npc_agent]
        
        self.assertTrue(len(history) >= 2) # Initial message + reply
        self.assertEqual(history[0]['content'], "Hello, Lydia.")
        self.assertEqual(history[0]['role'], "assistant") # From user_proxy perspective? No, usually 'content' and 'role'
        # user_proxy sends "Hello", so for user_proxy it's an outgoing message? 
        # Actually chat_messages stores messages as they appear in the chat.
        # Let's just check content presence.
        
        messages = [m['content'] for m in history]
        self.assertIn("Hello, Lydia.", messages)
        self.assertIn("I am sworn to carry your burdens, my Thane.", messages)

if __name__ == '__main__':
    unittest.main()
