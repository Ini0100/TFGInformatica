import json
import os
import logging

class GestorMemoriaAG2:
    """
    Componente encargado de serializar y deserializar el historial 
    de conversaciones de los agentes de AG2 para que sea persistente en el disco.
    """
    def __init__(self, carpeta_memorias="npc_conversations_db"):
        # Subimos a la raíz del proyecto para centralizar la carpeta
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ruta_carpeta = os.path.join(base_dir, carpeta_memorias)
        os.makedirs(self.ruta_carpeta, exist_ok=True)

    def _obtener_ruta_archivo(self, nombre_npc: str) -> str:
        # Reemplazamos espacios para evitar problemas de formato en el archivo
        nombre_limpio = nombre_npc.replace(" ", "_").lower()
        return os.path.join(self.ruta_carpeta, f"{nombre_limpio}_history.json")

    def guardar_conversacion(self, nombre_npc: str, agent_chat_history: list):
        """Guarda el historial de mensajes del agente en un archivo JSON."""
        ruta = self._obtener_ruta_archivo(nombre_npc)
        try:
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(agent_chat_history, f, ensure_ascii=False, indent=4)
            logging.info(f"[Memoria AG2] Historial guardado con éxito para {nombre_npc}.")
        except Exception as e:
            logging.error(f"[Memoria AG2] Error al guardar el historial de {nombre_npc}: {e}")

    def cargar_conversacion(self, nombre_npc: str) -> list:
        """Recuperates el historial de mensajes guardado. Devuelve una lista vacía si no existe."""
        ruta = self._obtener_ruta_archivo(nombre_npc)
        if not os.path.exists(ruta):
            logging.info(f"[Memoria AG2] No se encontró historial previo para {nombre_npc}. Iniciando conversación de cero.")
            return []
        
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                history = json.load(f)
            logging.info(f"[Memoria AG2] Historial de conversación recuperado ({len(history)} mensajes) para {nombre_npc}.")
            return history
        except Exception as e:
            logging.error(f"[Memoria AG2] Error al cargar el historial de {nombre_npc}: {e}")
            return []