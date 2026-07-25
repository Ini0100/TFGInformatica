import os
import chromadb

def inspeccionar_memoria():
    print("[*] Conectando a ChromaDB para inspección...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "npc_chroma_db")
    
    try:
        # Conectamos a la misma base de datos
        client = chromadb.PersistentClient(path=db_path)
        coleccion = client.get_collection(name="memoria_global_npcs")
        
        # Extraemos absolutamente todo: documentos y metadatos
        datos = coleccion.get(include=["metadatas", "documents"])
        
        if not datos or not datos['documents']:
            print("[-] La base de datos está vacía.")
            return
            
        # Agrupamos los fragmentos de texto por el nombre del NPC asignado
        memoria_por_npc = {}
        for i in range(len(datos['ids'])):
            doc = datos['documents'][i]
            meta = datos['metadatas'][i] if datos['metadatas'] else {}
            
            # Si no tiene el metadato npc_name, lo marcamos como Huérfano
            npc = meta.get("npc_name", "Sin asignar / Huérfano")
            
            if npc not in memoria_por_npc:
                memoria_por_npc[npc] = []
            memoria_por_npc[npc].append(doc)
            
        # Imprimimos el resumen por consola
        print("\n=== MAPA DE CONOCIMIENTO RAG POR NPC ===")
        for npc, documentos in memoria_por_npc.items():
            print(f"\n👤 NPC: {npc} (Total fragmentos inyectados: {len(documentos)})")
            print("-" * 60)
            for idx, doc in enumerate(documentos, 1):
                # Limpiamos saltos de línea y mostramos solo los primeros 80 caracteres
                preview = doc.replace('\n', ' ')[:80]
                print(f"  {idx}. {preview}...")
        print("\n========================================\n")
        
    except Exception as e:
        print(f"[!] Error al acceder a ChromaDB: {e}")

if __name__ == "__main__":
    inspeccionar_memoria()