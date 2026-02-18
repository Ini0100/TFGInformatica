# Guía de Integración con Google Gemini

Esta guía explica cómo configurar Mantella para utilizar los modelos Gemini de Google (como Gemini 1.5 Flash o Pro) como proveedor de LLM.

## Prerrequisitos

1.  Una cuenta de Google.
2.  Una API Key de Google AI Studio.

## Pasos para la Configuración

### 1. Obtener la API Key de Gemini

1.  Ve a [Google AI Studio](https://aistudio.google.com/).
2.  Haz clic en **"Get API key"**.
3.  Crea una clave API en un proyecto nuevo o existente.
4.  Copia la clave API (empieza por `AIza...`).

### 2. Configurar la API Key en Mantella

1.  Navega a la carpeta de instalación del mod Mantella (`.../Skyrim Special Edition/Data/Mantella...`).
2.  Abre el archivo `GPT_SECRET_KEY.txt`.
3.  Pega tu API Key de Gemini en este archivo y guarda los cambios. Asegúrate de que no haya espacios extra ni líneas adicionales.

### 3. Configurar Mantella para usar Gemini

Mantella permite conectar con cualquier servicio compatible con la API de OpenAI. Gemini ofrece esta compatibilidad.

1.  Abre la interfaz de configuración de Mantella (ejecutando `Mantella.exe` o a través del juego si usas la UI integrada).
2.  Ve a la sección **LLM (Large Language Model)**.
3.  En el campo **LLM Service**, selecciona o escribe manualmente la siguiente URL base:
    ```
    https://generativelanguage.googleapis.com/v1beta/openai/
    ```
    *Nota: Es posible que debas escribirlo manualmente si no aparece en la lista.*

4.  En el campo **Model**, escribe el nombre del modelo de Gemini que deseas usar. Algunos ejemplos comunes son:
    *   `gemini-1.5-flash` (Rápido y económico)
    *   `gemini-1.5-pro` (Más potente, mejor razonamiento)
    *   `gemini-2.0-flash-exp` (Experimental, muy rápido)

5.  Guarda la configuración.

### 4. Verificación

1.  Inicia Mantella y el juego.
2.  Intenta hablar con un NPC.
3.  Si todo está correcto, Mantella enviará las solicitudes a Google Gemini y recibirás respuesta.

## Solución de Problemas

*   **Error de conexión / API Key inválida:** Asegúrate de que la URL base termina exactamente en `/openai/` y que tu API Key es correcta.
*   **Modelo no encontrado:** Verifica que el nombre del modelo sea correcto (puedes consultarlos en la documentación de Google AI Studio).
*   **Timeouts:** Gemini suele ser rápido, pero si experimentas cortes, intenta aumentar el `Wait Time Buffer` en la configuración.

---

## Información Técnica (Para desarrolladores)

Mantella utiliza la librería `openai` de Python. La configuración anterior funciona porque Google ha implementado una interfaz compatible con OpenAI en su API.

Internamente, Mantella hace esto:
1.  Lee `GPT_SECRET_KEY.txt`.
2.  Inicializa el cliente `OpenAI` con la `base_url` proporcionada (`https://generativelanguage.googleapis.com/v1beta/openai/`).
3.  Envía las peticiones de chat completion usando el `model` especificado (`gemini-1.5-flash`).

No se requieren cambios en el código fuente de Mantella para esta integración básica.
