import os
import sys
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def cargar_documento_a_npc(ruta_archivo: str, nombre_npc: str):
    # nombre_npc debe coincidir exactamente con el nombre del personaje en Mantella:
    # si no coincide, el conocimiento se carga pero nunca se recupera.
    print(f"\n[+] Iniciando carga de conocimiento para: {nombre_npc}")

    if not os.path.exists(ruta_archivo):
        print(f"[!] Error: No se encuentra el archivo {ruta_archivo}")
        return
        
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_completo = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    parrafos = text_splitter.split_text(texto_completo)
    print(f"[*] El documento se ha dividido en {len(parrafos)} fragmentos.")

    # Mismo modelo de embeddings que en la recuperación (main_custom.py, GestorMemoriaAG2).
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "npc_chroma_db")

    chroma_client = chromadb.PersistentClient(path=db_path)
    coleccion = chroma_client.get_or_create_collection(
        name="memoria_global_npcs",
        embedding_function=emb_fn
    )

    # Se borra lo cargado antes para ese NPC, así el script es idempotente.
    print(f"[*] Purgando memoria antigua de {nombre_npc} para evitar duplicados...")
    coleccion.delete(where={"npc_name": nombre_npc})

    ids =[str(uuid.uuid4()) for _ in range(len(parrafos))]
    metadatas = [{"tipo": "documento_externo", "npc_name": nombre_npc} for _ in range(len(parrafos))]
    
    coleccion.add(
        documents=parrafos,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"[+] ¡ÉXITO! Base de datos actualizada para {nombre_npc}.")
    print(f"[*] Total de documentos actuales en la BBDD global: {coleccion.count()}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        archivo, npc_objetivo = sys.argv[1], sys.argv[2]
    else:
        # Valores por defecto si se ejecuta sin argumentos.
        archivo = "star_wars.txt"
        npc_objetivo = "Nazeem"
        print(f"[*] Uso: python cargar_documentos.py <archivo.txt> <NombreDelNPC>")
        print(f"[*] Sin argumentos, se usan los valores por defecto: {archivo} -> {npc_objetivo}")

    cargar_documento_a_npc(archivo, npc_objetivo)