import json
import os
import logging

class GestorMemoriaAG2:
    def __init__(self, base_dir="npc_conversations_db"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def obtener_ruta_archivo(self, npc_name):
        return os.path.join(self.base_dir, f"{npc_name}_memory.json")

    def cargar_conversacion(self, npc_name: str, emocion_actual: str) -> list:
        """
        Carga el historial. Si la emoción actual difiere de la guardada, 
        purga la memoria para evitar el sesgo de aprendizaje en contexto.
        """
        ruta = self.obtener_ruta_archivo(npc_name)
        if not os.path.exists(ruta):
            return []

        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Control de compatibilidad por si existen historiales antiguos (lista plana)
            if isinstance(data, list):
                logging.warning("[Memoria] Formato heredado detectado. Se purgará en el próximo guardado.")
                emocion_guardada = "desconocida"
                historial = data
            else:
                emocion_guardada = data.get("emocion", "neutral")
                historial = data.get("historial", [])

            # El núcleo de la solución: validación de estado emocional
            if emocion_guardada != emocion_actual:
                print(f"\n[!] Disrupción emocional detectada ({emocion_guardada} -> {emocion_actual}).")
                print(f"[!] Limpiando caché de memoria de {npc_name} para evitar arrastre cognitivo...")
                return [] 

            return historial
            
        except Exception as e:
            logging.error(f"[Memoria] Error al cargar el disco: {e}")
            return []

    def guardar_conversacion(self, npc_name: str, historial: list, emocion_actual: str):
        """
        Guarda el historial envolviéndolo en un diccionario de estado.
        """
        ruta = self.obtener_ruta_archivo(npc_name)
        
        # Estructura de guardado con metadatos
        data = {
            "emocion": emocion_actual,
            "historial": historial
        }
        
        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"[Memoria] Error crítico al escribir en disco: {e}")