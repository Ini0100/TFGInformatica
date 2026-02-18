import logging
import os
import chromadb
from src.remember.remembering import Remembering
from src.characters_manager import Characters
from src.llm.message_thread import message_thread
from src.llm.messages import UserMessage, AssistantMessage

class ChromaMemory(Remembering):
    def __init__(self, path="data/chroma_db"):
        self.path = path
        # Initialize client with persistent storage
        try:
            self.client = chromadb.PersistentClient(path=self.path)
            # Get or create the collection for NPC memories
            self.collection = self.client.get_or_create_collection(name="npc_memories")
            logging.info(f"ChromaDB initialized at {self.path}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def get_prompt_text(self, npcs_in_conversation: Characters, world_id: str) -> str:
        """
        Retrieves relevant memories from ChromaDB.
        """
        try:
            npc_names = [npc.name for npc in npcs_in_conversation.get_all_characters()]
            query_text = f"Interactions involving {', '.join(npc_names)}"
            
            # Construct where clause
            # We want to match existing world_id AND (memories containing any of the current NPCs)
            
            # If we have NPCs, add a filter for them
            if npc_names:
                if len(npc_names) == 1:
                     # Simple AND: world_id AND npc_names list contains X
                     where_clause = {
                         "$and": [
                             {"world_id": world_id},
                             {"npc_names": {"$contains": npc_names[0]}}
                         ]
                     }
                else:
                    # OR condition for multiple NPCs combined with world_id
                    or_conditions = [{"npc_names": {"$contains": name}} for name in npc_names]
                    where_clause = {
                        "$and": [
                            {"world_id": world_id},
                            {"$or": or_conditions}
                        ]
                    }
            else:
                # No NPCs, just filter by world_id
                where_clause = {"world_id": world_id}

            results = self.collection.query(
                query_texts=[query_text],
                n_results=5,
                where=where_clause
            )
            
            if results and results['documents']:
                memories = [doc for sublist in results['documents'] for doc in sublist]
                if memories:
                    return "Relevant past memories:\n" + "\n".join(f"- {m}" for m in memories)
            
            return ""
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")
            return ""

    def save_conversation_state(self, messages: message_thread, npcs_in_conversation: Characters, world_id: str, is_reload=False):
        """
        Saves the conversation messages to ChromaDB.
        """
        try:
            msgs_to_save = messages.get_talk_only()
            if not msgs_to_save:
                return

            ids = []
            documents = []
            metadatas = []
            
            import hashlib

            # Store as list of strings for better querying with $contains
            npc_names_list = [n.name for n in npcs_in_conversation.get_all_characters()]

            for msg in msgs_to_save:
                content_str = msg.get_formatted_content()
                if not content_str.strip():
                    continue

                # Create a deterministic ID
                # Use world_id + npc_names + content to ensure uniqueness per context if needed, 
                # but content hash + world_id is usually enough to dedup identical messages in same world
                content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
                msg_id = f"{world_id}_{content_hash}"
                
                role = "user" if isinstance(msg, UserMessage) else "assistant"
                
                ids.append(msg_id)
                documents.append(content_str)
                metadatas.append({
                    "world_id": world_id,
                    "role": role,
                    "npc_names": npc_names_list 
                })

            if ids:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logging.info(f"Saved {len(ids)} messages to ChromaDB")

        except Exception as e:
            logging.error(f"Error saving to ChromaDB: {e}")
