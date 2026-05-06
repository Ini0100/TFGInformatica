
import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

class MockConfigValues:
    def get_value(self, key):
        if key == "temperature": return 0.7
        return None

class MockConfig:
    def __init__(self):
        # Attributes present in ConfigLoader based on inspection
        self.llm_api = "https://generativelanguage.googleapis.com/v1beta/openai"
        self.llm = "gemini-flash-latest" # Matches properly loaded config
        self.llm_params = {"max_tokens": 250}
        self.definitions = MockConfigValues()

class MockCharacter:
    def __init__(self, name, bio):
        self.name = name
        self.bio = bio
        self.ref_id = "12345"

try:
    print("Importing AgentFactory...")
    from src.ag2.agent_factory import AgentFactory
    
    mock_char = MockCharacter("Lydia", "I am sworn to carry your burdens.")
    mock_config = MockConfig()
    
    print("Attempting to create agent with corrected config...")
    agent = AgentFactory.create_agent(mock_char, mock_config)
    
    print(f"Agent created successfully! Name: {agent.name}")
    print(f"Agent LLM Config: {agent.llm_config['config_list'][0]['model']}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
