import asyncio
import sys
import os
import logging
import time
import csv

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.my_custom_client import MyCustomClient
from src.config.definitions.llm_definitions import NarrationIndicatorsEnum
import tiktoken

# Mocks para simular el entorno de Mantella sin abrir el juego
class MockConfig:
    def __init__(self):
        self.llm_api = "http://127.0.0.1:11434/v1"
        self.llm = "gemma4:e2b"
        self.llm_params = {"temperature": 0.8, "max_tokens": 800, "stop": ["<TERMINAR>"]}
        self.custom_token_count = 4096
        self.vision_enabled = False
        self.advanced_actions_enabled = False
        self.narration_indicators = NarrationIndicatorsEnum.PARANTHESES

class MockMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    def get_openai_message(self) -> dict:
        return {"role": self.role, "content": self.content}
    def get_formatted_content(self) -> str:
        return self.content

class MockMessageThread:
    def __init__(self, messages):
        self.messages = messages
    def get_openai_messages(self):
        return [m.get_openai_message() for m in self.messages]
    def get_last_message(self):
        return self.messages[-1]
    def add_message(self, message):
        self.messages.append(message)
    def add_system_message(self, text):
        self.messages.append(MockMessage("system", text))

BATERIA_PRUEBAS = [
    ("Nazeem", "¿Qué fue la Guerra de los Clones?", "star_wars"),
    ("Nazeem", "Háblame del Canciller Supremo Palpatine.", "star_wars"),
    ("Farengar", "¿Quién es Tony Stark?", "avengers"),
    ("Farengar", "Cuéntame quién es Thor Odinson.", "avengers"),
    ("Nazeem", "¿Cuál es tu plato favorito para desayunar?", "control_sin_memoria"),
]

_encoding = tiktoken.get_encoding("cl100k_base")

def contar_tokens(texto: str) -> int:
    return len(_encoding.encode(texto or ""))


def instrumentar_cliente(cliente: MyCustomClient, metricas: dict):
    """Envuelve las funciones internas de MyCustomClient con cronómetros para
    medir cada fase del pipeline RAG + HyDE de forma independiente."""

    hyde_original = cliente._generar_documento_hipotetico
    async def hyde_instrumentado(pregunta, npc_actual):
        t0 = time.perf_counter()
        resultado = await hyde_original(pregunta, npc_actual)
        metricas['t_hyde'] = time.perf_counter() - t0
        return resultado
    cliente._generar_documento_hipotetico = hyde_instrumentado

    query_original = cliente.coleccion_memoria.query
    def query_instrumentada(*args, **kwargs):
        t0 = time.perf_counter()
        resultado = query_original(*args, **kwargs)
        metricas['t_retrieval'] = time.perf_counter() - t0
        docs = resultado.get('documents') if resultado else None
        metricas['docs_recuperados'] = len(docs[0]) if docs else 0
        return resultado
    cliente.coleccion_memoria.query = query_instrumentada

    predict_original = cliente.cross_encoder.predict
    def predict_instrumentado(pares):
        t0 = time.perf_counter()
        scores = predict_original(pares)
        metricas['t_rerank'] = time.perf_counter() - t0
        metricas['score_top1'] = float(max(scores)) if len(scores) else None
        return scores
    cliente.cross_encoder.predict = predict_instrumentado


async def iniciar_consola_custom_client():
    print("\n[+] Iniciando MyCustomClient (modo conversación manual)...")
    config = MockConfig()

    cliente = MyCustomClient(
        config=config,
        secret_key_file="GPT_SECRET_KEY.txt",
        image_secret_key_file="IMAGE_SECRET_KEY.txt"
    )

    print(f"[+] Documentos disponibles en la coleccion de memoria: {cliente.coleccion_memoria.count()}")

    npc = input("\n¿Con qué NPC quieres hablar? (ej. Nazeem, Farengar): ").strip() or "Farengar"
    thread = MockMessageThread([MockMessage("system", f"Eres {npc}.")])

    print(f"\n=== CONVERSACIÓN CON {npc.upper()} LISTA (escribe 'salir' para terminar) ===")

    try:
        while True:
            texto_usuario = input("\nTú: ").strip()
            if texto_usuario.lower() in ['salir', 'exit', 'quit']:
                break
            if not texto_usuario:
                continue

            thread.add_message(MockMessage("user", texto_usuario))

            respuesta_completa = ""
            try:
                async for chunk_type, chunk_data in cliente.streaming_call(thread, is_multi_npc=False):
                    if chunk_type == "content" and chunk_data:
                        respuesta_completa += chunk_data
                        print(chunk_data, end="", flush=True)
            except Exception as e:
                print(f"\n[Error durante la generación]: {e}")
            print()

            thread.add_message(MockMessage("assistant", respuesta_completa.strip()))
    finally:
        print("\n[+] Conversación finalizada.")


async def ejecutar_bateria():
    print("\n[+] Iniciando MyCustomClient (modo bateria de pruebas automatizada)...")
    config = MockConfig()

    cliente = MyCustomClient(
        config=config,
        secret_key_file="GPT_SECRET_KEY.txt",
        image_secret_key_file="IMAGE_SECRET_KEY.txt"
    )

    print(f"[+] Documentos disponibles en la coleccion de memoria: {cliente.coleccion_memoria.count()}")

    resultados = []

    for npc, pregunta, tema in BATERIA_PRUEBAS:
        metricas_turno = {}
        instrumentar_cliente(cliente, metricas_turno)

        historial = [MockMessage("system", f"Eres {npc}."), MockMessage("user", pregunta)]
        thread = MockMessageThread(historial)

        print(f"\n--- [{npc}] {pregunta} ---")
        respuesta_completa = ""
        t_inicio = time.perf_counter()
        try:
            async for chunk_type, chunk_data in cliente.streaming_call(thread, is_multi_npc=False):
                if chunk_type == "content" and chunk_data:
                    respuesta_completa += chunk_data
                    print(chunk_data, end="", flush=True)
        except Exception as e:
            print(f"\n[Error durante la generación]: {e}")
        t_total = time.perf_counter() - t_inicio
        print()

        tokens_pregunta = contar_tokens(pregunta)
        tokens_respuesta = contar_tokens(respuesta_completa)

        fila = {
            "npc": npc,
            "pregunta": pregunta,
            "tema_esperado": tema,
            "t_hyde_s": round(metricas_turno.get('t_hyde', 0.0), 3),
            "t_retrieval_s": round(metricas_turno.get('t_retrieval', 0.0), 3),
            "t_rerank_s": round(metricas_turno.get('t_rerank', 0.0), 3),
            "t_total_s": round(t_total, 3),
            "docs_recuperados": metricas_turno.get('docs_recuperados', 0),
            "score_top1": round(metricas_turno['score_top1'], 3) if metricas_turno.get('score_top1') is not None else None,
            "tokens_pregunta": tokens_pregunta,
            "tokens_respuesta": tokens_respuesta,
            "respuesta": respuesta_completa.strip(),
        }
        resultados.append(fila)
        await asyncio.sleep(1)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_test_custom_client.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\n[+] Resultados guardados en {csv_path}")

    print("\n=== RESUMEN ===")
    header = f"{'NPC':<10}{'t_total(s)':<12}{'t_hyde(s)':<11}{'t_retr(s)':<11}{'t_rerank(s)':<13}{'score':<8}{'tok_in':<8}{'tok_out':<8}"
    print(header)
    print("-" * len(header))
    for r in resultados:
        print(f"{r['npc']:<10}{r['t_total_s']:<12}{r['t_hyde_s']:<11}{r['t_retrieval_s']:<11}{r['t_rerank_s']:<13}{str(r['score_top1']):<8}{r['tokens_pregunta']:<8}{r['tokens_respuesta']:<8}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("¿Cómo quieres ejecutar este test?")
    print("  [1] Modo interactivo (escribir preguntas manualmente)")
    print("  [2] Batería de pruebas automática (preguntas fijas + métricas + tabla de resultados)")
    modo = input("Elige una opción [1/2]: ").strip()

    if modo == "1":
        asyncio.run(iniciar_consola_custom_client())
    else:
        asyncio.run(ejecutar_bateria())
