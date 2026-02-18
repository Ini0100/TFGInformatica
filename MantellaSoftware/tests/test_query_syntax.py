
import chromadb
import shutil
import os
import logging

def test_query():
    path = "data/chroma_debug"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    client = chromadb.PersistentClient(path=path)
    collection = client.create_collection("debug_collection")
    
    # Add data
    collection.add(
        ids=["1", "2"],
        documents=["Hello", "Hi there"],
        metadatas=[
            {"world_id": "world1", "npc_name": "Alice, Bob"},
            {"world_id": "world1", "npc_name": "Charlie"}
        ]
    )
    
    print(f"ChromaDB Version: {chromadb.__version__}")

    # Add data with list?
    try:
        collection.add(
            ids=["3"],
            documents=["Greetings"],
            metadatas=[{"world_id": "world1", "npc_names_list": ["Alice", "Bob"]}] # List support?
        )
        print("List metadata storage: Success")
    except Exception as e:
        print(f"List metadata storage: Failed ({e})")

    # Test queries
    print("--- Test 1: Simple world_id ---")
    res = collection.query(query_texts=["Hello"], n_results=5, where={"world_id": "world1"})
    print(res['ids'])

    print("\n--- Test 2: Contains substring Alice (on string field) ---")
    res = collection.query(query_texts=["Hello"], n_results=5, where={"npc_name": {"$contains": "Alice"}})
    print(res['ids'])
    
    print("\n--- Test 4: Query contained in List? ---")
    # If list storage worked, try querying
    try:
        # Chroma way to query list membership: "field": "value" (implicit contains)
        res = collection.query(query_texts=["Greetings"], n_results=5, where={"npc_names_list": "Alice"})
        print(f"List query 'Alice': {res['ids']}")
        
        # Or using $contains?
        res = collection.query(query_texts=["Greetings"], n_results=5, where={"npc_names_list": {"$contains": "Alice"}})
        print(f"List query $contains 'Alice': {res['ids']}")
    except Exception as e:
        print(f"List query error: {e}")

if __name__ == "__main__":
    test_query()
