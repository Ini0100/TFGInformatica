import asyncio
import sys
import os
import time
import csv
import logging

# Configuración básica
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autogen import AssistantAgent, UserProxyAgent
from src.llm.gestor_memoria_ag2 import GestorMemoriaAG2
import tiktoken

NOMBRE_NPC = "Farengar"
SYSTEM_MESSAGE = "Eres Farengar Fuego Secreto, mago de Carrera Blanca. TUS RESPUESTAS DEBEN SER BREVES."

_encoding = tiktoken.get_encoding("cl100k_base")

def contar_tokens(texto: str) -> int:
    return len(_encoding.encode(texto or ""))

def tokens_de_historial(historial: list) -> int:
    """Tokens acumulados de TODO el historial que AG2 reenvía en cada turno
    (a diferencia de test_custom_client, aquí no hay RAG/re-ranking que
    resuma el contexto: AG2 reenvía la conversación completa cada vez)."""
    return sum(contar_tokens(m.get("content", "")) for m in historial)


def crear_llm_config() -> dict:
    return {
        "config_list": [{
            "model": "gemma4:e2b",
            "api_key": "abc123",
            "base_url": "http://127.0.0.1:11434/v1",
            "price": [0.0, 0.0],
        }],
        "temperature": 0.8,
        "timeout": 120,
        # Sin esto, AutoGen usa cache_seed=41 por defecto y cachea en disco
        # (.cache/41/): con memoria persistente entre "sesiones" (ver
        # GestorMemoriaAG2), una misma pregunta con el mismo historial podría
        # devolver una respuesta cacheada de una ejecución anterior en vez de
        # generar una nueva, contaminando la reproducibilidad de la batería.
        "cache_seed": None,
    }


def crear_agentes(llm_config: dict):
    farengar_agent = AssistantAgent(
        name=NOMBRE_NPC,
        system_message=SYSTEM_MESSAGE,
        llm_config=llm_config,
    )
    player_proxy = UserProxyAgent(
        name="Jugador",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    return farengar_agent, player_proxy


# ---------------------------------------------------------------------------
# Modo interactivo (comportamiento original del script: entrada manual por
# teclado, con persistencia de memoria entre ejecuciones vía JSON).
# ---------------------------------------------------------------------------
def iniciar_consola_ag2_1vs1():
    print("\n[+] Iniciando AG2 (1 vs 1) con MEMORIA PERSISTENTE (modo interactivo)...")

    llm_config = crear_llm_config()

    gestor = GestorMemoriaAG2()
    historial_previo = gestor.cargar_conversacion(NOMBRE_NPC, "neutral")

    farengar_agent, player_proxy = crear_agentes(llm_config)

    if historial_previo:
        print(f"[*] Cargando {len(historial_previo)} mensajes anteriores...")
        farengar_agent.chat_messages[player_proxy] = historial_previo

    print("\n=== CONSOLA LISTA ===")

    try:
        while True:
            texto_usuario = input("\nTú: ")
            if texto_usuario.lower() in ['salir', 'exit', 'quit']:
                break

            player_proxy.initiate_chat(
                farengar_agent,
                message=texto_usuario,
                clear_history=False,
                silent=False
            )

    finally:
        historial_actualizado = farengar_agent.chat_messages.get(player_proxy, [])
        gestor.guardar_conversacion(NOMBRE_NPC, historial_actualizado, "neutral")
        print(f"\n[+] Memoria guardada ({len(historial_actualizado)} mensajes).")


# ---------------------------------------------------------------------------
# Modo batería automática: mide métricas específicas de la orquestación AG2
# (no de la recuperación RAG, que ya se valida en test_custom_client.py):
#
#   - t_turno_s: latencia de cada llamada a initiate_chat().
#   - mensajes_historial: longitud del historial de chat tras cada turno,
#     para observar cómo crece el contexto que AG2 reenvía en cada turno.
#   - tokens_contexto_enviado: tokens estimados de TODO el historial que se
#     reenvía a la API en ese turno (AG2 no resume ni filtra el contexto
#     como sí hace el RAG de MyCustomClient, así que este coste crece con
#     cada turno).
#   - tokens_respuesta: tokens de la respuesta generada en ese turno.
#   - memoria_recordada_en_sesion: verificación de que el agente recuerda,
#     dentro de la MISMA sesión, un dato mencionado en un turno anterior.
#   - persistencia_entre_sesiones: verificación de que, al recargar el
#     historial guardado en disco (simulando reabrir el juego), el número
#     de mensajes recuperados coincide con los guardados.
#   - purga_por_disrupcion_emocional: verificación de que, si la emoción
#     declarada no coincide con la guardada, GestorMemoriaAG2 purga el
#     historial en lugar de arrastrar contexto obsoleto.
# ---------------------------------------------------------------------------
BATERIA_TURNOS = [
    "Hola, me llamo Godofredo y quiero aprender sobre hechizos de fuego.",
    "¿Qué opinas de la magia de hielo en comparación con la de fuego?",
    "¿Recuerdas cómo me llamo?",
]


def ejecutar_bateria_ag2():
    print("\n[+] Iniciando AG2 (1 vs 1) con MEMORIA PERSISTENTE (modo batería automática)...")
    llm_config = crear_llm_config()

    # Empezamos la "sesión 1" desde un historial limpio para que la prueba
    # sea reproducible, guardando lo anterior si existía.
    gestor = GestorMemoriaAG2()
    resultados = []

    farengar_agent, player_proxy = crear_agentes(llm_config)

    recuerda_nombre = None
    persistencia_ok = None
    purga_ok = None
    historial_sesion_1 = []

    print(f"[*] Sesión 1: ejecutando {len(BATERIA_TURNOS)} turnos fijos...")
    for i, texto_usuario in enumerate(BATERIA_TURNOS, start=1):
        try:
            t0 = time.perf_counter()
            player_proxy.initiate_chat(
                farengar_agent,
                message=texto_usuario,
                clear_history=False,
                silent=True,
            )
            t_turno = time.perf_counter() - t0
        except Exception as e:
            print(f"\n[!] La API del LLM falló en el turno {i} ({e}). Se detiene la batería y se "
                  f"guardan los resultados obtenidos hasta ahora.")
            break

        historial = farengar_agent.chat_messages.get(player_proxy, [])
        respuesta = historial[-1]["content"] if historial else ""
        print(f"\n--- Turno {i}: {texto_usuario}")
        print(f"NPC: {respuesta}")

        resultados.append({
            "fase": f"sesion_1_turno_{i}",
            "mensaje": texto_usuario,
            "respuesta": respuesta,
            "t_turno_s": round(t_turno, 3),
            "mensajes_historial": len(historial),
            "tokens_contexto_enviado": tokens_de_historial(historial[:-1]),
            "tokens_respuesta": contar_tokens(respuesta),
        })

    sesion_1_completa = len(resultados) == len(BATERIA_TURNOS)

    if resultados:
        # Verificación de memoria a corto plazo DENTRO de la misma sesión:
        # el último turno ejecutado pregunta explícitamente por un dato (el
        ultima_respuesta = resultados[-1]["respuesta"]
        recuerda_nombre = "godofredo" in ultima_respuesta.lower()
        print(f"\n[Verificación] ¿El agente recuerda el nombre mencionado en turnos anteriores? {'SÍ' if recuerda_nombre else 'NO'}")

        historial_sesion_1 = farengar_agent.chat_messages.get(player_proxy, [])
        gestor.guardar_conversacion(NOMBRE_NPC, historial_sesion_1, "neutral")
        print(f"[*] Memoria de la sesión 1 guardada ({len(historial_sesion_1)} mensajes).")

    if sesion_1_completa:
        # --- Sesión 2: nueva instancia de agente (simula reabrir el juego) ---
        print("\n[*] Sesión 2: recargando memoria desde disco con la MISMA emoción (neutral)...")
        farengar_agent_2, player_proxy_2 = crear_agentes(llm_config)
        historial_recargado = gestor.cargar_conversacion(NOMBRE_NPC, "neutral")
        persistencia_ok = len(historial_recargado) == len(historial_sesion_1)
        print(f"[Verificación] Mensajes guardados: {len(historial_sesion_1)} | Mensajes recargados: {len(historial_recargado)} | ¿Coinciden? {'SÍ' if persistencia_ok else 'NO'}")

        if historial_recargado:
            farengar_agent_2.chat_messages[player_proxy_2] = historial_recargado
            pregunta_continuidad = "¿Sigues recordando de qué hablamos?"
            try:
                t0 = time.perf_counter()
                player_proxy_2.initiate_chat(
                    farengar_agent_2,
                    message=pregunta_continuidad,
                    clear_history=False,
                    silent=True,
                )
                t_turno = time.perf_counter() - t0
                historial_2 = farengar_agent_2.chat_messages.get(player_proxy_2, [])
                respuesta_2 = historial_2[-1]["content"] if historial_2 else ""
                print(f"\n--- Turno sesión 2: {pregunta_continuidad}")
                print(f"NPC: {respuesta_2}")
                resultados.append({
                    "fase": "sesion_2_continuidad",
                    "mensaje": pregunta_continuidad,
                    "respuesta": respuesta_2,
                    "t_turno_s": round(t_turno, 3),
                    "mensajes_historial": len(historial_2),
                    "tokens_contexto_enviado": tokens_de_historial(historial_2[:-1]),
                    "tokens_respuesta": contar_tokens(respuesta_2),
                })
            except Exception as e:
                print(f"\n[!] La API del LLM falló en el turno de continuidad ({e}). Se omite esta verificación.")

        # --- Sub-prueba: purga por disrupción emocional (no requiere LLM) ---
        # Si la emoción declarada difiere de la guardada, GestorMemoriaAG2 debe
        # descartar el historial en vez de arrastrar contexto obsoleto.
        print("\n[*] Sub-prueba: recargando memoria con una emoción DISTINTA a la guardada (enfadado)...")
        historial_tras_disrupcion = gestor.cargar_conversacion(NOMBRE_NPC, "enfadado")
        purga_ok = len(historial_tras_disrupcion) == 0
        print(f"[Verificación] Mensajes tras cambio de emoción: {len(historial_tras_disrupcion)} | ¿Se purgó correctamente? {'SÍ' if purga_ok else 'NO'}")

        # Restauramos el fichero de memoria a su estado neutral para no dejar
        # el purgado como estado final en disco.
        gestor.guardar_conversacion(NOMBRE_NPC, historial_sesion_1, "neutral")
    else:
        print("\n[*] Se omiten las verificaciones de persistencia entre sesiones y de disrupción "
              "emocional porque la sesión 1 no se completó.")

    # --- Guardar resultados (aunque la batería se haya interrumpido a medias) ---
    if resultados:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_test_ag2_client.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\n[+] Resultados por turno guardados en {csv_path}")
    else:
        print("\n[!] No se obtuvo ningún resultado que guardar.")

    def fmt(valor):
        if valor is None:
            return "N/D (no ejecutado)"
        return "OK" if valor else "FALLO"

    print("\n=== RESUMEN DE VERIFICACIONES ===")
    print(f"{'Verificación':<55}{'Resultado'}")
    print("-" * 70)
    print(f"{'Memoria de corto plazo dentro de la misma sesión':<55}{fmt(recuerda_nombre)}")
    print(f"{'Persistencia del historial entre sesiones (JSON)':<55}{fmt(persistencia_ok)}")
    print(f"{'Purga de memoria por disrupción emocional':<55}{fmt(purga_ok)}")

    if resultados:
        print("\n=== RESUMEN POR TURNO ===")
        header = f"{'Fase':<22}{'t_turno(s)':<12}{'msgs_hist':<11}{'tok_ctx':<10}{'tok_resp':<10}"
        print(header)
        print("-" * len(header))
        for r in resultados:
            print(f"{r['fase']:<22}{r['t_turno_s']:<12}{r['mensajes_historial']:<11}{r['tokens_contexto_enviado']:<10}{r['tokens_respuesta']:<10}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("¿Cómo quieres ejecutar este test?")
    print("  [1] Modo interactivo (escribir preguntas manualmente, como antes)")
    print("  [2] Batería de pruebas automática (preguntas fijas + métricas + tabla de resultados)")
    modo = input("Elige una opción [1/2]: ").strip()

    if modo == "2":
        ejecutar_bateria_ag2()
    else:
        iniciar_consola_ag2_1vs1()
