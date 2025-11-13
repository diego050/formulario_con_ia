import pandas as pd
import json
import time
import google.generativeai as genai
from typing import List, Dict
import asyncio


# --- CONFIGURACIÓN Y CONSTANTES ---

# Cargar la configuración de scoring desde el JSON
try:
    with open("scoring_config.json", "r") as f:
        config_data = json.load(f)
    SCORING_CONFIG = config_data.get("SCORING_CATEGORIES", {})
    print("✅ Configuración de scoring cargada exitosamente.")
except FileNotFoundError:
    print("❌ ERROR: El archivo 'scoring_config.json' no se encontró.")
    SCORING_CONFIG = {}
except json.JSONDecodeError:
    print("❌ ERROR: El archivo 'scoring_config.json' no es un JSON válido.")
    SCORING_CONFIG = {}


STATUS_HIERARCHY = {
    "Investment": {"score": 10, "description": "Éxito máximo. La startup recibió inversión. Este es el objetivo final."},
    "Investment committee": {"score": 9, "description": "Etapa final. Muy prometedora, pasó a comité de inversión."},
    "Interview UV": {"score": 8, "description": "Etapa avanzada. Pasó los filtros iniciales y tuvo una entrevista formal."},
    "Reference Checks": {"score": 7, "description": "Etapa de validación. Interesante, se están verificando referencias."},
    "Reviewing": {"score": 5, "description": "Etapa media. Superó el screening inicial y está en revisión activa."},
    "Screening": {"score": 4, "description": "Etapa temprana. Apenas se está evaluando si cumple lo mínimo."},
    "Backlog": {"score": 3, "description": "En espera. No es una prioridad ahora, pero no se ha descartado."},
    "Rechazo con feed": {"score": 1, "description": "Rechazada. No cumple los criterios, aunque se le dio feedback."},
}

# --- LÓGICA DE SCORING CON IA (PROMPT ACTUALIZADO) ---

def get_llm_dimensional_scoring(startup_data: str, full_historical_portfolio: str, thesis_context: str) -> dict:
    """
    Función de scoring que utiliza el contexto completo del portafolio histórico en cada llamada.
    """
    default_response = {
        "dimensional_scores": {category: 0 for category in SCORING_CONFIG.keys()},
        "qualitative_analysis": { "project_thesis": "Error", "problem": "Error", "solution": "Error", "key_metrics": "Error", "founding_team": "Error", "market_and_competition": "Error" },
        "score_justification": { "equipo": "Error", "tesis_utec": "Error", "oportunidad": "Error", "validacion": "Error" }
    }

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Usamos el modelo correcto
        status_hierarchy_prompt = "\n".join(f'- {s} (Nivel {d["score"]}/10): {d["description"]}' for s, d in STATUS_HIERARCHY.items())
        
        prompt = f"""
            Eres un analista de Venture Capital de clase mundial en UTEC Ventures. Tu tarea es realizar un análisis profundo de una startup candidata, usando un rico contexto histórico y estratégico para informar tu evaluación. Devuelve un informe completo en formato JSON.

            **CONTEXTO ESTRATÉGICO Y DATOS HISTÓRICOS:**

            1.  **Tesis de Inversión de UTEC Ventures (Nuestra Filosofía):**
                ```
                {thesis_context}
                ```

            2.  **Portafolio Histórico Completo (Nuestra Experiencia):**
                A continuación se presenta una lista en formato JSON de todas las startups en las que hemos invertido o analizado en el pasado, incluyendo sus datos y su 'Status' final. Usa estos datos para entender patrones, qué sectores nos interesan, y cómo se ve el éxito vs. el fracaso en nuestro fondo.
                ```json
                {full_historical_portfolio}
                ```
            
            3.  **Jerarquía de Status (Cómo medimos el éxito):**
                {status_hierarchy_prompt}

            **TAREA:**

            Ahora, analiza la siguiente startup candidata basándote en TODO el contexto proporcionado:
            
            **Datos de la Startup a Analizar:**
            ```json
            {startup_data}
            ```

            **INSTRUCCIONES:**
            Completa la siguiente estructura JSON. Tus análisis y puntajes deben reflejar una comprensión profunda de nuestra tesis y los patrones de nuestro portafolio histórico. Sé conciso pero informativo (1-2 frases). Los puntajes deben ser de 0 a 100.

            **Formato de Salida JSON (OBLIGATORIO Y ÚNICO):**
            ```json
            {{
                "dimensional_scores": {{
                    "equipo": <0-100>, "producto": <0-100>, "tesis_utec": <0-100>, 
                    "oportunidad": <0-100>, "validacion": <0-100>
                }},
                "qualitative_analysis": {{
                    "project_thesis": "Resume la tesis principal de la startup.",
                    "problem": "Describe el problema.",
                    "solution": "Describe la solución.",
                    "key_metrics": "Lista las métricas clave.",
                    "founding_team": "Describe al equipo fundador.",
                    "market_and_competition": "Resume el mercado y competencia."
                }},
                "score_justification": {{
                    "equipo": "Justifica el puntaje de 'equipo'.",
                    "tesis_utec": "Justifica el puntaje de 'tesis_utec' alineado a la Tesis de Inversión y el portafolio.",
                    "oportunidad": "Justifica el puntaje de 'oportunidad' (Market Cap, etc.) comparándolo con casos pasados.",
                    "validacion": "Justifica el puntaje de 'validacion' (Logros, tracción) en base a lo que históricamente ha funcionado."
                }}
            }}
            ```
            """
        
        print(" -> Enviando prompt masivo a la IA...")
        response = model.generate_content(prompt)
        
        text_response = response.text.strip()
        json_start = text_response.find('{')
        json_end = text_response.rfind('}') + 1

        if json_start != -1 and json_end != -1:
            json_text = text_response[json_start:json_end]
            return json.loads(json_text)
        else:
            print(" -> ADVERTENCIA: No se encontró un JSON válido en la respuesta de la IA.")
            return default_response

    except Exception as e:
        print(f"\n !!! ERROR AL PROCESAR RESPUESTA DE LA IA: {e} !!!")
        return default_response


# --- FUNCIÓN PRINCIPAL DEL PROCESO DE SCORING ---

async def run_scoring_loop(df_to_score: pd.DataFrame, df_context: pd.DataFrame, thesis_context: str) -> List[Dict]:
    results = []
    total_startups = len(df_to_score)

    print("Convirtiendo el portafolio histórico completo a JSON para el contexto...")
    full_portfolio_context_json = df_context.to_json(orient='records', indent=2)

    for index, row in df_to_score.iterrows():
        startup_name = row.get('Nombre de la startup') or row.get('Nombre', f'Fila {index + 1}')
        print(f"\n[ {index + 1} / {total_startups} ] Procesando: '{startup_name}'...")

        startup_json = row.where(pd.notna(row), None).to_json()
        
        llm_result = get_llm_dimensional_scoring(
            startup_data=startup_json,
            full_historical_portfolio=full_portfolio_context_json,
            thesis_context=thesis_context
        )

        dimensional_scores = llm_result.get("dimensional_scores", {})
        
        final_score = 0
        if SCORING_CONFIG:
            final_score = sum(
                (dimensional_scores.get(cat, 0) or 0) * details.get("peso", 0)
                for cat, details in SCORING_CONFIG.items()
            )
        
        original_data = row.where(pd.notna(row), None).to_dict()
        result_row = {
            **original_data,
            **llm_result,
            "final_weighted_score": round(final_score, 2)
        }
        
        print(f" -> Análisis completo para '{startup_name}'. Score: {result_row['final_weighted_score']}")
        results.append(result_row)
        
        print(" -> Esperando 6 segundos para no exceder el límite de RPM...")
        time.sleep(6)

    return results


async def run_single_scoring(startup_dict: dict, df_context: pd.DataFrame, thesis_context: str) -> Dict:
    """
    Procesa el scoring para una única startup recibida como un diccionario.
    """
    startup_name = startup_dict.get('Nombre de la startup') or startup_dict.get('Nombre', 'Startup sin nombre')
    print(f"\n[ Re-análisis ] Procesando: '{startup_name}'...")

    # Convertir el portafolio histórico a JSON para el contexto
    full_portfolio_context_json = df_context.to_json(orient='records', indent=2)
    startup_json = json.dumps(startup_dict)

    # Llamar a la misma función de scoring de la IA que ya tenemos
    llm_result = get_llm_dimensional_scoring(
        startup_data=startup_json,
        full_historical_portfolio=full_portfolio_context_json,
        thesis_context=thesis_context
    )

    dimensional_scores = llm_result.get("dimensional_scores", {})
    
    # Calcular el nuevo puntaje final
    final_score = 0
    if SCORING_CONFIG:
        final_score = sum(
            (dimensional_scores.get(cat, 0) or 0) * details.get("peso", 0)
            for cat, details in SCORING_CONFIG.items()
        )
    
    # Combinar los datos originales con el nuevo resultado de la IA
    result_row = {
        **startup_dict,
        **llm_result,
        "final_weighted_score": round(final_score, 2)
    }
    
    print(f" -> Re-análisis completo para '{startup_name}'. Nuevo Score: {result_row['final_weighted_score']}")
    return result_row

async def run_scoring_loop_stream(df_to_score: pd.DataFrame, df_context: pd.DataFrame, thesis_context: str):
    """
    Versión generadora del bucle de scoring.
    Usa 'yield' para transmitir cada resultado en lugar de devolver una lista.
    """
    total_startups = len(df_to_score)
    print("Convirtiendo el portafolio histórico a JSON para el contexto de streaming...")
    full_portfolio_context_json = df_context.to_json(orient='records', indent=2)

    for index, row in df_to_score.iterrows():
        startup_name = row.get('Nombre de la startup') or row.get('Nombre', f'Fila {index + 1}')
        print(f"\n[ Stream / {index + 1} de {total_startups} ] Procesando: '{startup_name}'...")

        startup_json = row.where(pd.notna(row), None).to_json()
        
        llm_result = get_llm_dimensional_scoring(
            startup_data=startup_json,
            full_historical_portfolio=full_portfolio_context_json,
            thesis_context=thesis_context
        )

        dimensional_scores = llm_result.get("dimensional_scores", {})
        
        final_score = 0
        if SCORING_CONFIG:
            final_score = sum(
                (dimensional_scores.get(cat, 0) or 0) * details.get("peso", 0)
                for cat, details in SCORING_CONFIG.items()
            )
        
        original_data = row.where(pd.notna(row), None).to_dict()
        result_row = {
            **original_data,
            **llm_result,
            "final_weighted_score": round(final_score, 2)
        }
        
        # El formato "data: ...\n\n" es requerido por el protocolo Server-Sent Events
        yield f"data: {json.dumps(result_row)}\n\n"
        
        print(f" -> Stream enviado para '{startup_name}'. Esperando 6s...")
        await asyncio.sleep(6)