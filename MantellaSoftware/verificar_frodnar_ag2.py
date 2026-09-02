import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2

# Reproduce exactamente lo que hace AG2Conversation en producción: instanciar
# GestorMemoriaAG2() sin argumentos y pedirle el contexto de un NPC, tal y
# como hace AgentFactory.create_agent() al construir el system_message.
gestor = GestorMemoriaAG2()
print(f"[+] ¿Colección ChromaDB conectada?: {gestor.coleccion_chroma is not None}")

contexto = gestor.obtener_contexto("Frodnar")
print(f"\n[+] Contexto recuperado para Frodnar ({len(contexto)} caracteres):\n")
print(contexto if contexto else "(vacío)")

contexto_otro = gestor.obtener_contexto("NPC_Que_No_Existe")
print(f"\n[+] Contexto para un NPC sin datos (debe estar vacío): '{contexto_otro}'")
