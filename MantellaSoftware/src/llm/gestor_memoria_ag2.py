import json
import os
import logging
import chromadb
from chromadb.utils import embedding_functions

class GestorMemoriaAG2:
    def __init__(self, coleccion_chroma=None, base_dir="npc_conversations_db"):
        # Si no se pasa una colección explícita, nos conectamos directamente a
        # la misma base de conocimiento que usa MyCustomClient/HyDE
        # ("npc_chroma_db" / "memoria_global_npcs", la que rellenan
        # cargar_documentos.py y cargar_documento_frodnar.py). Antes, cuando se
        # llamaba sin argumentos (como hace AG2Conversation en producción), el
        # gestor se quedaba sin ChromaDB en absoluto -> obtener_contexto()
        # devolvía siempre "" y ningún conocimiento RAG llegaba jamás a los
        # agentes AG2, aunque estuviera correctamente cargado en la base de
        # datos.
        if coleccion_chroma is None:
            try:
                base_dir_proyecto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                db_path = os.path.join(base_dir_proyecto, "npc_chroma_db")
                emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
                chroma_client = chromadb.PersistentClient(path=db_path)
                coleccion_chroma = chroma_client.get_or_create_collection(
                    name="memoria_global_npcs",
                    embedding_function=emb_fn
                )
            except Exception as e:
                logging.error(f"[Memoria] No se pudo conectar automáticamente a la base de conocimiento RAG: {e}")
                coleccion_chroma = None

        self.coleccion_chroma = coleccion_chroma
        self.base_dir = base_dir

        os.makedirs(self.base_dir, exist_ok=True)

        if self.coleccion_chroma:
            logging.info("[Memoria] Gestor enlazado correctamente con ChromaDB.")
        else:
            logging.warning("[Memoria] Gestor iniciado sin ChromaDB (solo modo JSON local).")

    def obtener_ruta_archivo(self, npc_name):
        return os.path.join(self.base_dir, f"{npc_name}_memory.json")

    def obtener_contexto(self, npc_name: str) -> str:
        """
        Recupera la memoria a largo plazo (RAG) del NPC desde ChromaDB
        para inyectarla en la inicialización del Agente AG2.
        """
        if not self.coleccion_chroma:
            return ""
            
        try:
            # La base de conocimiento RAG ("memoria_global_npcs") etiqueta cada
            # fragmento con el campo "npc_name" (string exacto, no una lista),
            # tal y como lo escriben cargar_documentos.py y
            # cargar_documento_frodnar.py, y como lo lee MyCustomClient. Un
            # filtro "npc_names"/$contains (pensado para el esquema de
            # ChromaMemory, que es una colección distinta) nunca encontraba
            # nada aquí.
            resultados = self.coleccion_chroma.get(
                where={"npc_name": npc_name}
            )
            
            if resultados and resultados.get('documents') and len(resultados['documents']) > 0:
                logging.info(f"[Memoria RAG] Contexto cargado con éxito para {npc_name}.")
                return "\n".join(resultados['documents'])
            
            return ""
        except Exception as e:
            logging.error(f"[Memoria RAG] Error al extraer el contexto de {npc_name}: {e}")
            return ""

    def cargar_conversacion(self, npc_name: str, emocion_actual: str) -> list:
        ruta = self.obtener_ruta_archivo(npc_name)
        if not os.path.exists(ruta):
            return []

        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                logging.warning("[Memoria] Formato heredado detectado. Se purgará en el próximo guardado.")
                emocion_guardada = "desconocida"
                historial = data
            else:
                emocion_guardada = data.get("emocion", "neutral")
                historial = data.get("historial", [])

            if emocion_guardada != emocion_actual:
                print(f"\n[!] Disrupción emocional detectada ({emocion_guardada} -> {emocion_actual}).")
                print(f"[!] Limpiando caché de memoria de {npc_name} para evitar arrastre cognitivo...")
                return [] 

            return historial
            
        except Exception as e:
            logging.error(f"[Memoria] Error al cargar el disco: {e}")
            return []

    def guardar_conversacion(self, npc_name: str, historial: list, emocion_actual: str):
        ruta = self.obtener_ruta_archivo(npc_name)
        
        data = {
            "emocion": emocion_actual,
            "historial": historial
        }
        
        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"[Memoria] Error crítico al escribir en disco: {e}")