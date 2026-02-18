
import os
from autogen import AssistantAgent, UserProxyAgent

def test_ag2_basic():
    # Configuration for the agents
    # We will use the config list from environment or hardcoded for now just to test import and object creation
    # In a real scenario, we'd load this from Mantella's config
    
    # Note: This requires an API key. For this basic test, we might mock it or just check instantiation.
    # To fully run it, we need a valid config_list. 
    # For now, let's just test we can import and instantiate the classes, 
    # which confirms the library is installed and working.
    
    print("Initializing Agents...")
    
    try:
        assistant = AssistantAgent(
            name="Assistant",
            llm_config=False, # Disable LLM for this basic smoke test to avoid needing keys right now
        )

        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": "tests/ag2_work", "use_docker": False}
        )
        
        print("Agents initialized successfully.")
        print(f"Assistant: {assistant.name}")
        print(f"User Proxy: {user_proxy.name}")
        
        # If we had a key, we could do:
        # user_proxy.initiate_chat(assistant, message="Hello!")
        
    except Exception as e:
        print(f"Failed to initialize agents: {e}")
        raise

if __name__ == "__main__":
    test_ag2_basic()
