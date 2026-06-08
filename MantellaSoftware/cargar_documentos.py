import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importamos el registro para validar nombres de forma estricta
from src.llm.npc_emotions import REGISTRO_NPCS

def cargar_documento_a_npc(ruta_archivo: str, nombre_npc: str):
    # 1. VALIDACIÓN ESTRICTA: Evita errores tipográficos (ej. "Farengar Fuego Secreto")
    nombres_validos = [data["nombre"] for data in REGISTRO_NPCS.values()]
    if nombre_npc not in nombres_validos:
        print(f"[!] ERROR CRÍTICO: '{nombre_npc}' no es un NPC válido en tu sistema.")
        print(f"[*] Nombres permitidos: {nombres_validos}")
        return

    print(f"\n[+] Iniciando carga de conocimiento para: {nombre_npc}")
    
    # 2. Leer el archivo
    if not os.path.exists(ruta_archivo):
        print(f"[!] Error: No se encuentra el archivo {ruta_archivo}")
        return
        
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_completo = f.read()

    # 3. Fragmentación Recursiva
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    parrafos = text_splitter.split_text(texto_completo)
    print(f"[*] El documento se ha dividido en {len(parrafos)} fragmentos.")

    # 4. Conectar a ChromaDB (Alineado estrictamente con test_integrated_rag_ag2.py)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Ruta absoluta local uniforme
    db_path = os.path.join(base_dir, "npc_chroma_db")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    coleccion = chroma_client.get_or_create_collection(
        name="memoria_global_npcs", 
        embedding_function=emb_fn
    )

    # 5. IDEMPOTENCIA: Limpiar registros anteriores del NPC antes de insertar
    # Esto evita la duplicación infinita si ejecutas el script varias veces.
    print(f"[*] Purgando memoria antigua de {nombre_npc} para evitar duplicados...")
    coleccion.delete(where={"npc_name": nombre_npc})

    # 6. Inyectar en la base de datos
    ids = [str(uuid.uuid4()) for _ in range(len(parrafos))]
    metadatas = [{"tipo": "documento_externo", "npc_name": nombre_npc} for _ in range(len(parrafos))]
    
    coleccion.add(
        documents=parrafos,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"[+] ¡ÉXITO! Base de datos actualizada para {nombre_npc}.")
    print(f"[*] Total de documentos actuales en la BBDD global: {coleccion.count()}")

if __name__ == "__main__":
    # --- CONFIGURA TU INGESTA AQUÍ ---
    # Asegúrate de borrar la carpeta 'npc_chroma_db' vieja antes de ejecutar esto por primera vez.
    
    archivo = "star_wars.txt" 
    npc_objetivo = "Nazeem" 
    
    cargar_documento_a_npc(archivo, npc_objetivo)