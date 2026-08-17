import json
import os
import logging

class GestorMemoriaAG2:
    def __init__(self, coleccion_chroma=None, base_dir="npc_conversations_db"):
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
            # ChromaMemory guarda los nombres de NPCs en el campo "npc_names" (lista),
            # por lo que hay que usar $contains en vez de comparar un campo "npc_name" que no existe.
            resultados = self.coleccion_chroma.get(
                where={"npc_names": {"$contains": npc_name}}
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