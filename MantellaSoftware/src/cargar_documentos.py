import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def cargar_documento_a_npc(ruta_archivo: str, nombre_npc: str):
    print(f"\n[+] Iniciando carga de conocimiento para: {nombre_npc}")
    
    # 1. Leer el archivo
    if not os.path.exists(ruta_archivo):
        print(f"[!] Error: No se encuentra el archivo {ruta_archivo}")
        return
        
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_completo = f.read()

    # 2. Fragmentación Recursiva con Solapamiento (NUEVO)
    # chunk_size: Tamaño máximo del fragmento
    # chunk_overlap: Caracteres que se solapan con el fragmento anterior
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Jerarquía de corte
    )
    
    parrafos = text_splitter.split_text(texto_completo)
    print(f"[*] El documento se ha dividido dinámicamente en {len(parrafos)} fragmentos (con solapamiento).")

    # 3. Conectar a ChromaDB (Con corrección de rutas)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    base_dir = os.path.dirname(os.path.dirname(__file__)) 
    db_path = os.path.join(base_dir, "npc_chroma_db")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    coleccion = chroma_client.get_or_create_collection(
        name="memoria_global_npcs", 
        embedding_function=emb_fn
    )

    # 4. Inyectar en la base de datos
    ids = [str(uuid.uuid4()) for _ in range(len(parrafos))]
    metadatas = [{"tipo": "documento_externo", "npc_name": nombre_npc} for _ in range(len(parrafos))]
    
    coleccion.add(
        documents=parrafos,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"[+] ¡ÉXITO! Conocimiento inyectado en el cerebro de {nombre_npc}.")
    print(f"[*] Total de documentos en la BBDD global: {coleccion.count()}")

if __name__ == "__main__":
    # AQUÍ CONFIGURAS LO QUE QUIERES SUBIR
    archivo_apuntes = "apuntes_vengadores.txt" 
    cargar_documento_a_npc(archivo_apuntes, "Nazeem")