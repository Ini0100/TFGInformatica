import logging
import os
import re
from typing import AsyncGenerator
from openai import AsyncOpenAI

import chromadb
from chromadb.utils import embedding_functions

from src.llm.llm_client import LLMClient
from src.llm.message_thread import message_thread
from src.llm.messages import SystemMessage, UserMessage

from sentence_transformers import CrossEncoder
import numpy as np
import asyncio

class MyCustomClient(LLMClient):
    """
    Cliente LLM con Sistema RAG de Grado Producción (ChromaDB + HyDE).
    Versión LITE: Optimizada para máxima estabilidad (Sin dependencias PyTorch/CrossEncoder).
    """

    NPC_SYSTEM_INJECTION = (
        "REGLAS CRÍTICAS DE ROLEPLAY:\n"
        "1. Eres un personaje en el mundo de Skyrim (Tamriel). NUNCA reconozcas que eres una IA.\n"
        "2. Responde en primera persona y mantén un tono acorde a tu personaje.\n"
        "3. Eres un experto en ciertos temas. Si se te proporciona INFORMACIÓN DE REFERENCIA, úsala para enseñar o responder al jugador, integrándola en tu diálogo de forma natural."
    )

    def __init__(self, config, secret_key_file: str, image_secret_key_file: str, function_secret_key_file: str = None) -> None:
        
        # Dentro del __init__ de MyCustomClient
        logging.info("[Re-Ranker] Cargando modelo CrossEncoder en VRAM (CUDA)...")

        # Instanciamos el modelo forzando su carga en la tarjeta gráfica
        self.cross_encoder = CrossEncoder(
            'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
            device='cuda'
        )
        
        super().__init__(config, secret_key_file, image_secret_key_file, function_secret_key_file)
        logging.info("========= MyCustomClient (RAG + HyDE Lite) INICIALIZADO =========")

        logging.info("[Base de Conocimiento] Cargando ChromaDB...")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        db_path = os.path.join(base_dir, "npc_chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        self.coleccion_memoria = self.chroma_client.get_or_create_collection(
            name="memoria_global_npcs", 
            embedding_function=self.emb_fn
        )

        # 2. En el __init__ de MyCustomClient, inicializa el modelo CrossEncoder
        logging.info("[Re-Ranker] Cargando modelo CrossEncoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

        api_key = ""
        ruta_key = os.path.join(base_dir, secret_key_file)
        if os.path.exists(ruta_key):
            with open(ruta_key, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
                
        self.hyde_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai"
        )

    def _extraer_nombre_npc(self, messages) -> str:
        try:
            if hasattr(messages, 'messages') and isinstance(messages.messages, list):
                for msg in messages.messages:
                    if getattr(msg, 'role', '') == 'system' or (hasattr(msg, 'get_openai_message') and msg.get_openai_message().get('role') == 'system'):
                        contenido = getattr(msg, 'content', '') if hasattr(msg, 'content') else msg.get_formatted_content()
                        match = re.search(r'(?:You are|Eres)\s+([^,\.]+)', contenido, re.IGNORECASE)
                        if match:
                            return match.group(1).strip()
        except Exception as e:
            logging.error(f"[Base de Conocimiento] Error extrayendo NPC: {e}")
        return "Desconocido"

    async def _generar_documento_hipotetico(self, pregunta: str, npc_actual: str) -> str:
        try:
            logging.info(f"[HyDE] Iniciando transformación de consulta para: '{pregunta}'")
            prompt_hyde = (
                f"Escribe un párrafo enciclopédico, técnico y muy breve (máximo 2 líneas) que responda directamente "
                f"a la siguiente consulta académica: '{pregunta}'.\n"
                f"Proporciona datos fácticos, nombres propios y definiciones objetivas. "
                f"NUNCA uses introducciones, saludos, comentarios ni lenguaje de rol."
            )
            
            response = await self.hyde_client.chat.completions.create(
                model="gemini-flash-latest",
                messages=[{"role": "user", "content": prompt_hyde}],
                temperature=0.1, 
                max_tokens=120
            )
            doc_hipotetico = response.choices[0].message.content.strip()
            logging.info(f"[HyDE] Documento imaginado generado:\n > {doc_hipotetico}")
            return doc_hipotetico
        except Exception as e:
            logging.error(f"[HyDE] Error al generar documento: {e}")
            return pregunta

    async def streaming_call(self, messages, is_multi_npc: bool, tools: list[dict] = None) -> AsyncGenerator[tuple[str, str | list] | None, None]:
        texto_usuario = ""
        npc_actual = self._extraer_nombre_npc(messages)

        if hasattr(messages, 'get_last_message'):
            last_msg = messages.get_last_message()
            if hasattr(last_msg, 'get_openai_message'):
                if last_msg.get_openai_message().get('role', '') == 'user' and hasattr(last_msg, 'get_formatted_content'):
                    texto_usuario = last_msg.get_formatted_content()
                
            if texto_usuario:
                total_docs = self.coleccion_memoria.count()
                
                if total_docs > 0:
                    texto_busqueda = await self._generar_documento_hipotetico(texto_usuario, npc_actual)
                    
                    # FASE 1: Recuperación Amplia (Hasta 15 documentos)
                    resultados = self.coleccion_memoria.query(
                        query_texts=[texto_busqueda], 
                        n_results=min(15, total_docs), 
                        where={"npc_name": npc_actual} if npc_actual != "Desconocido" else None
                    )
                    
                    if resultados and resultados['documents'] and len(resultados['documents'][0]) > 0:
                        documentos_candidatos = resultados['documents'][0]

                        # FASE 2: Re-Ranking con Cross-Encoder
                        # Preparamos los pares (Pregunta Original, Documento Candidato)
                        pares = [[texto_usuario, doc] for doc in documentos_candidatos]
        
                        # El modelo evalúa la relevancia cruzada de cada par
                        scores = self.cross_encoder.predict(pares)
        
                        # Ordenamos los índices de mayor a menor puntuación usando numpy
                        mejores_indices = np.argsort(scores)[::-1]
        
                        # Seleccionamos estrictamente los 3 mejores
                        top_3_indices = mejores_indices[:3]
                        mejores_documentos = [documentos_candidatos[i] for i in top_3_indices]
        
                        documentos_texto = "\n\n".join(mejores_documentos)
                        
                        if hasattr(messages, 'messages') and isinstance(messages.messages, list) and len(messages.messages) > 0:
                            messages.messages.insert(-1, msg_rag)
                        elif hasattr(messages, 'add_system_message'):
                            messages.add_system_message(inyeccion_rag)

        async for chunk_type, chunk_data in super().streaming_call(messages, is_multi_npc, tools):
            yield chunk_type, chunk_data
