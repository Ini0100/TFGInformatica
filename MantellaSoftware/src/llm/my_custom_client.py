import logging
import os
import uuid
import re
from typing import AsyncGenerator

import chromadb
from chromadb.utils import embedding_functions

from src.llm.llm_client import LLMClient
from src.llm.message_thread import message_thread
from src.llm.messages import SystemMessage, UserMessage

class MyCustomClient(LLMClient):
    """
    Cliente LLM personalizado con Memoria a Largo Plazo (ChromaDB).
    Intercepta las llamadas, aísla memorias por NPC usando metadatos y guarda los nuevos.
    """

    NPC_SYSTEM_INJECTION = (
        "REGLAS CRÍTICAS DE ROLEPLAY:\n"
        "1. Eres un personaje real en el mundo de Skyrim (Tamriel). NUNCA reconozcas que eres una IA.\n"
        "2. Responde siempre en primera persona y mantén un tono acorde a la época medieval fantástica.\n"
        "3. Tus respuestas deben ser naturales, fluidas y concisas.\n"
        "4. Expresa tus acciones físicas o gestos entre asteriscos (*suspiro*, *desenfunda la espada*)."
    )

    def __init__(self, config, secret_key_file: str, image_secret_key_file: str, function_secret_key_file: str = None) -> None:
        super().__init__(config, secret_key_file, image_secret_key_file, function_secret_key_file)
        logging.info("========= MyCustomClient INICIALIZADO =========")

        logging.info("[Memoria] Cargando modelo de Embeddings y ChromaDB...")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        db_path = os.path.join(os.path.dirname(__file__), "npc_chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        self.coleccion_memoria = self.chroma_client.get_or_create_collection(
            name="memoria_global_npcs", 
            embedding_function=self.emb_fn
        )
        logging.info(f"[Memoria] Base de datos lista. Documentos actuales guardados: {self.coleccion_memoria.count()}")

    def _extraer_nombre_npc(self, messages) -> str:
        """
        Analiza el historial de mensajes para deducir con quién estamos hablando.
        Busca patrones como 'You are [Nombre]' o 'Eres [Nombre]'.
        """
        try:
            if hasattr(messages, 'messages') and isinstance(messages.messages, list):
                for msg in messages.messages:
                    # Buscamos en los mensajes del sistema
                    if getattr(msg, 'role', '') == 'system' or (hasattr(msg, 'get_openai_message') and msg.get_openai_message().get('role') == 'system'):
                        contenido = getattr(msg, 'content', '') if hasattr(msg, 'content') else msg.get_formatted_content()
                        
                        # Regex para capturar el nombre después de "Eres" o "You are", parando en la primera coma o punto
                        match = re.search(r'(?:You are|Eres)\s+([^,\.]+)', contenido, re.IGNORECASE)
                        if match:
                            nombre = match.group(1).strip()
                            return nombre
        except Exception as e:
            logging.error(f"[Memoria] Error extrayendo nombre del NPC: {e}")
        
        return "Desconocido" # Fallback por si la regex falla

    async def streaming_call(self, messages, is_multi_npc: bool, tools: list[dict] = None) -> AsyncGenerator[tuple[str, str | list] | None, None]:
        texto_usuario = ""
        
        # Obtenemos el nombre del NPC actual
        npc_actual = self._extraer_nombre_npc(messages)
        logging.info(f"[Memoria] Identidad del NPC detectada: {npc_actual}")

        # 1. Recuperar el texto del usuario de forma segura
        if hasattr(messages, 'get_last_message'):
            last_msg = messages.get_last_message()
            if hasattr(last_msg, 'get_openai_message'):
                rol = last_msg.get_openai_message().get('role', '')
                if rol == 'user' and hasattr(last_msg, 'get_formatted_content'):
                    texto_usuario = last_msg.get_formatted_content()
                
            # 2. Búsqueda e inyección controlada (FILTRANDO POR NPC)
            if texto_usuario:
                total_docs = self.coleccion_memoria.count()
                if total_docs > 0:
                    n_res = min(5, total_docs)
                    
                    # AQUÍ ESTÁ LA MAGIA: Filtramos (where) para que solo busque recuerdos de ESTE NPC
                    resultados = self.coleccion_memoria.query(
                        query_texts=[texto_usuario],
                        n_results=n_res,
                        where={"npc_name": npc_actual} if npc_actual != "Desconocido" else None
                    )
                    
                    if resultados and resultados['documents'] and len(resultados['documents'][0]) > 0:
                        recuerdos_texto = "\n".join(resultados['documents'][0])
                        inyeccion_memoria = (
                            "ATENCIÓN AL SISTEMA: A continuación se muestran fragmentos literales de conversaciones "
                            "pasadas que has tenido con este jugador. DEBES usar esta información para responder a la pregunta actual. "
                            "NUNCA digas que has olvidado algo que está escrito en estos recuerdos.\n\n"
                            f"RECUERDOS RECUPERADOS:\n{recuerdos_texto}"
                        )
                        
                        msg_memoria = SystemMessage(inyeccion_memoria, config=None)
                        if hasattr(messages, 'messages') and isinstance(messages.messages, list) and len(messages.messages) > 0:
                            messages.messages.insert(-1, msg_memoria)
                        else:
                            if hasattr(messages, 'add_message'):
                                messages.add_message(msg_memoria)
                            elif hasattr(messages, 'add_system_message'):
                                messages.add_system_message(inyeccion_memoria)
                                
                        logging.info(f"[Memoria] ÉXITO: {len(resultados['documents'][0])} recuerdos recuperados para {npc_actual}.")
                else:
                    logging.info("[Memoria] INFO: La base de datos está vacía en este momento.")

            # Inyección de Roleplay general
            msg_roleplay = SystemMessage(self.NPC_SYSTEM_INJECTION, config=None)
            if hasattr(messages, 'messages') and isinstance(messages.messages, list) and len(messages.messages) > 0:
                messages.messages.insert(0, msg_roleplay)

        # 3. Llamada al LLM padre
        full_response_parts: list[str] = []
        async for chunk_type, chunk_data in super().streaming_call(messages, is_multi_npc, tools):
            if chunk_type == "content" and chunk_data:
                full_response_parts.append(chunk_data)
            yield chunk_type, chunk_data

        # 4. Almacenamiento comprobado (GUARDANDO METADATOS)
        full_response = "".join(full_response_parts)
        if texto_usuario and full_response:
            interaccion_a_guardar = f"Jugador: {texto_usuario} | {npc_actual}: {full_response}"
            
            # Guardamos el recuerdo ASOCIADO al NPC actual
            self.coleccion_memoria.add(
                documents=[interaccion_a_guardar],
                metadatas=[{"tipo": "dialogo", "npc_name": npc_actual}], 
                ids=[str(uuid.uuid4())]
            )
            logging.info(f"[Memoria] GUARDADO: Recuerdo asignado a {npc_actual}. Total DB: {self.coleccion_memoria.count()}")
