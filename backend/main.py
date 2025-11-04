import pandas as pd
import io
import json
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import time

# --- Jerarquía y significado de cada Status para la IA ---
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

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

if not API_KEY:
    print("ERROR: La variable de entorno GOOGLE_API_KEY no se encontró.")
else:
    print(f"API Key cargada exitosamente. Comienza con: {API_KEY[:4]}...")

with open("scoring_config.json", "r") as f:
    config_data = json.load(f)
    SCORING_CONFIG = config_data["SCORING_CATEGORIES"]

FEEDBACK_LOG_FILE = "feedback_log.csv"

app = FastAPI(
    title="Deal Flow AI API v2",
    description="API para puntuar postulaciones de startups y analizar datos históricos."
)

origins = [ "http://localhost:5173", "http://127.0.0.1:5173" ]
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

def analyze_portfolio(df_portfolio: pd.DataFrame) -> str:
    print("Analizando el portafolio para extraer contexto...")
    industry_col = 'Sector'
    if industry_col in df_portfolio.columns:
        top_industries = df_portfolio[industry_col].value_counts().head(3).index.tolist()
        return f"Las industrias clave son: {', '.join(top_industries)}."
    return "No se pudo determinar un patrón de industrias en el portafolio."

def get_feedback_examples(df_portfolio: pd.DataFrame) -> str:
    print("    -> Buscando ejemplos de contraste (éxito vs. fracaso) en el portafolio...")
    if 'Status' not in df_portfolio.columns:
        return "No se pudo crear ejemplos por falta de la columna 'Status'."
    df_portfolio_clean = df_portfolio.dropna(subset=['Status']).copy()
    best_example, worst_example = None, None
    for status in ["Investment", "Investment committee"]:
        examples = df_portfolio_clean[df_portfolio_clean['Status'] == status]
        if not examples.empty:
            best_example = examples.iloc[0]; break
    rejected_examples = df_portfolio_clean[df_portfolio_clean['Status'].str.contains('Rechazo', na=False)]
    if not rejected_examples.empty:
        worst_example = rejected_examples.iloc[0]
    if best_example is not None and worst_example is not None:
        return f"""... (Ejemplos de contraste) ..."""
    return "No se encontraron ejemplos claros."

async def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_content = await pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return "No se pudo leer el documento de contexto."

def get_llm_dimensional_scoring(startup_data: str, feedback_examples: str, thesis_context: str, portfolio_context: str) -> dict:
    default_response = {
        "dimensional_scores": {category: 0 for category in SCORING_CONFIG},
        "qualitative_analysis": { "project_thesis": "Error", "problem": "Error", "solution": "Error", "key_metrics": "Error", "founding_team": "Error", "market_and_competition": "Error" },
        "score_justification": { "equipo": "Error", "tesis_utec": "Error", "oportunidad": "Error", "validacion": "Error" }
    }
    if not API_KEY: return default_response
    model = genai.GenerativeModel('gemini-2.5-flash')
    status_hierarchy_prompt = "\n".join(f'- {s} (Nivel {d["score"]}/10): {d["description"]}' for s, d in STATUS_HIERARCHY.items())
    prompt = f"""
    Eres un analista de Venture Capital extremadamente diligente en UTEC Ventures. Tu tarea es analizar una startup y devolver un informe completo en formato JSON.
    **CONTEXTO:** 1. Jerarquía de Status: {status_hierarchy_prompt} 2. Tesis de Inversión: {thesis_context} 3. Ejemplos: {feedback_examples} 4. Datos de la Startup: {startup_data}
    **INSTRUCCIONES:** Basado en TODO el contexto, completa la siguiente estructura JSON. Sé conciso pero informativo (1-2 frases).
    **Formato de Salida JSON (OBLIGATORIO Y ÚNICO):**
    {{
      "dimensional_scores": {{"equipo": <0-100>, "producto": <0-100>, "tesis_utec": <0-100>, "oportunidad": <0-100>, "validacion": <0-100>}},
      "qualitative_analysis": {{"project_thesis": "Resume la tesis principal de la startup.", "problem": "Describe el problema.", "solution": "Describe la solución.", "key_metrics": "Lista las métricas clave.", "founding_team": "Describe al equipo fundador.", "market_and_competition": "Resume el mercado y competencia."}},
      "score_justification": {{"equipo": "Justifica el puntaje de 'equipo'.", "tesis_utec": "Justifica el puntaje de 'tesis_utec'.", "oportunidad": "Justifica el puntaje de 'oportunidad' (Market Cap).", "validacion": "Justifica el puntaje de 'validacion' (Logros)."}}
    }}
    """
    try:
        print("    -> Enviando prompt completo a la IA...")
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        start, end = json_text.find('{'), json_text.rfind('}') + 1
        if start != -1 and end != -1: return json.loads(json_text[start:end])
        else: print("    -> ADVERTENCIA: No se encontró un JSON válido."); return default_response
    except Exception as e:
        print(f"\n    !!! ERROR AL PROCESAR RESPUESTA COMPLEJA: {e} !!!"); return default_response

def save_to_excel_incrementally(result_data: dict, output_filename: str):
    try:
        df_new_row = pd.json_normalize(result_data, sep='_') # Usamos '_' como separador
        if not os.path.exists(output_filename):
            df_new_row.to_excel(output_filename, index=False, engine='openpyxl')
        else:
            df_existing = pd.read_excel(output_filename, engine='openpyxl')
            df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
            df_combined.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"    -> Guardado exitoso en Excel.")
    except Exception as e:
        print(f"\n    !!! ADVERTENCIA: No se pudo guardar en Excel: {e} !!!")

# --- ENDPOINTS ---
async def run_scoring_loop(df_to_score, df_context, thesis_context, output_filename):
    portfolio_context = analyze_portfolio(df_context)
    feedback_examples = get_feedback_examples(df_context)
    results = []
    total_startups = len(df_to_score)

    # --- NUEVA LÓGICA DE REANUDACIÓN ---
    completed_startups = set()
    key_column = 'Nombre de la startup' # Asegúrate que este es el nombre correcto de la columna

    try:
        if os.path.exists(output_filename):
            print(f"Archivo de resultados existente encontrado: '{output_filename}'. Intentando reanudar...")
            df_existing = pd.read_excel(output_filename, engine='openpyxl')

            # Verificamos si la columna clave existe en el archivo guardado
            # Usamos .get('Nombre', key_column) para compatibilidad con nombres antiguos
            if df_existing.columns.str.contains('Nombre').any():
                 # Buscamos la columna que contenga "Nombre" para identificarla.
                 actual_key_column = next((col for col in df_existing.columns if 'Nombre' in col), None)
                 if actual_key_column:
                    completed_startups = set(df_existing[actual_key_column].dropna())
                    print(f"Se encontraron {len(completed_startups)} startups ya procesadas. Se omitirán.")

    except Exception as e:
        print(f"Advertencia: No se pudo leer el archivo existente para reanudar. Se procesará desde el inicio si es necesario. Error: {e}")
    # --- FIN DE LA LÓGICA DE REANUDACIÓN ---

    for index, row in df_to_score.iterrows():
        # Usamos .get para evitar errores si la columna no existe en alguna fila
        startup_name = row.get(key_column) or row.get('Nombre', f'Fila {index + 1}')

        # --- NUEVO: Comprobación para omitir startups ya procesadas ---
        if startup_name in completed_startups:
            print(f"[ {index + 1} / {total_startups} ] Omitiendo: '{startup_name}' (ya procesado).")
            continue  # Salta a la siguiente iteración del bucle

        print(f"\n[ {index + 1} / {total_startups} ] Procesando: '{startup_name}'...")
        llm_result = get_llm_dimensional_scoring(row.to_json(), feedback_examples, thesis_context, portfolio_context)
        
        dimensional_scores = llm_result.get("dimensional_scores", {})
        
        final_score = sum((dimensional_scores.get(c) or 0) * d["peso"] for c, d in SCORING_CONFIG.items())
        
        original_data = row.where(pd.notna(row), None).to_dict()
        result_row = {**original_data, **llm_result, "final_weighted_score": round(final_score, 2)}
        
        save_to_excel_incrementally(result_row, output_filename)
        results.append(result_row)
        
        print(f" -> Análisis completo para '{startup_name}'. Score: {result_row['final_weighted_score']}")
        time.sleep(1.1)
        
    return results


@app.post("/api/process-and-score")
async def process_and_score(applications_file: UploadFile=File(...), portfolio_file: UploadFile=File(...), context_pdf: Optional[UploadFile]=File(None)):
    output_filename = "scored_applications.xlsx"
    if os.path.exists(output_filename): os.remove(output_filename)
    apps_content = await applications_file.read(); df_apps = pd.read_csv(io.BytesIO(apps_content))
    portfolio_content = await portfolio_file.read(); df_portfolio = pd.read_csv(io.BytesIO(portfolio_content))
    thesis_context = await extract_text_from_pdf(context_pdf) if context_pdf else "No se proporcionó tesis."
    print(f"\n--- INICIANDO SCORING DE NUEVAS POSTULACIONES ---")
    results = await run_scoring_loop(df_apps, df_portfolio, thesis_context, output_filename)
    print("\n--- PROCESO DE SCORING DE POSTULACIONES FINALIZADO ---")
    return results

@app.post("/api/score-historical-data")
async def score_historical_data(historical_data_file: UploadFile=File(...), context_pdf: Optional[UploadFile]=File(None)):
    output_filename = "scored_historical_data.xlsx"
    #if os.path.exists(output_filename): os.remove(output_filename)
    historical_content = await historical_data_file.read(); df_to_score = pd.read_csv(io.BytesIO(historical_content))
    thesis_context = await extract_text_from_pdf(context_pdf) if context_pdf else "No se proporcionó tesis."
    print(f"\n--- INICIANDO SCORING DE DATA HISTÓRICA ---")
    results = await run_scoring_loop(df_to_score, df_to_score.copy(), thesis_context, output_filename)
    print("\n--- PROCESO DE SCORING HISTÓRICO FINALIZADO ---")
    return results

@app.post("/api/submit-feedback")
async def submit_feedback(data: dict = Body(...)):
    try:
        df_feedback = pd.DataFrame([data]); df_feedback.to_csv(FEEDBACK_LOG_FILE, mode='a', header=not os.path.exists(FEEDBACK_LOG_FILE), index=False)
        return {"status": "Feedback guardado con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el feedback: {e}")
