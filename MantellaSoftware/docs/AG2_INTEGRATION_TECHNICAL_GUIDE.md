# AG2 (AutoGen) Integration Technical Guide

## Overview
This document details the integration of [ag2](https://github.com/ag2ai/ag2) (formerly AutoGen) into Mantella. The goal is to evolve Mantella from a single-LLM-call architecture to a multi-agent system where NPCs are autonomous agents capable of conversing with each other and the player.

## Architecture

### Core Components

1.  **`src/ag2/agent_factory.py`**
    -   **Responsibility**: Converts Mantella `Character` objects into AG2 `AssistantAgent` instances.
    -   **Logic**:
        -   Extracts character bio/name.
        -   Configures LLM settings (API Key, Model, Base URL) based on `config.ini`.
        -   Sanitizes names for AG2 compatibility.

2.  **`src/ag2/mantella_user_proxy.py`**
    -   **Responsibility**: Represents the Player in the AG2 ecosystem.
    -   **Logic**:
        -   Extends `UserProxyAgent`.
        -    Acts as a bridge for Mantella's STT (Speech-to-Text) and Input system.
        -   (Future) Will intercept `get_human_input` to return transcribed text from the game.


3.  **`src/ag2/ag2_client.py`**
    -   **Responsibility**: Adapter class that implements `AIClient` interface but wraps AG2 agents.
    -   **Logic**:
        -   Implements `streaming_call` to trigger `agent.generate_reply()`.
        -   Yields the agent's reply prefixed with `SpeakerName: ` so `ChangeCharacterParser` can switch the active speaker in `ChatManager`.
        -   Allows reusing Mantella's existing TTS and subtitle pipeline.

4.  **`src/ag2/ag2_conversation.py`**
    -   **Responsibility**: The main controller for agent-based conversations.
    -   **Logic**:
        -   Inherits from `Conversation`.
        -   Initializes AG2 agents for all NPCs.
        -   Creates an `AG2Client` with these agents.
        -   Instantiates a new `ChatManager` using `AG2Client`.
        -   Uses the standard `Conversation` loop, which now calls `AG2Client` instead of `LLMClient`.

## Implementation Steps

### Phase 1: Infrastructure (Completed)
-   Installed `ag2[openai]`.
-   Created `AgentFactory` and `MantellaUserProxy`.
-   Verified basic conversation flow in `tests/test_ag2_conversation.py`.

### Phase 2: Game Loop Integration (In Progress)
-   **Adapter Pattern**: Create `AG2Conversation` that mimics the interface of the original `Conversation` class so `GameStateManager` can switch between them easily.
-   **State Management**: Ensure AG2's internal chat history is synced with Mantella's memory requirements (or rely on AG2's history).
-   **Output Handling**: AG2 agents "print" to console/logger. We need to capture these replies and send them to the `TextToSpeech` and `SkyrimSender` modules.

## Key Challenges & Solutions

### 1. Capturing Agent Output
**Challenge**: AG2 agents handle their own output. Mantella needs to capture the generated text to generate audio (TTS) and display subtitles.
**Solution**:
-   Register a `reply_func` or hook into the `AssistantAgent`'s message handler.
-   Or, use a custom `GroupChatManager` that emits events on every message.

### 2. Player Input Timing
**Challenge**: Mantella's `Conversation.loop()` polls for player input. AG2's `UserProxyAgent.get_human_input()` is blocking.
**Solution**:
-   Override `get_human_input` in `MantellaUserProxy` to check Mantella's specific input queue (non-blocking or timeout-based) to keep the game loop alive.

## Usage
To enable AG2 (once integrated):
1.  Ensure `ag2` is installed.

2.  Enable `use_ag2 = True` in `config.ini` (under `[Mantella]`).
3.  Restart Mantella.
