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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse 

# Rutas a tus archivos fijos (hardcodeados)
HISTORICAL_PORTFOLIO_PATH = "historical_data.csv" # Tu base de datos histórica puntuada
CONTEXT_PDF_PATH = "proyecto_preprofesional.pdf" # Tu PDF de contexto

# Variables globales para almacenar los datos cargados
df_portfolio_context: pd.DataFrame = None
thesis_context_text: str = ""

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

origins = ["*"]
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

@app.on_event("startup")
async def startup_event():
    """
    Esta función se ejecuta una sola vez cuando la aplicación FastAPI se inicia.
    Carga los datos de contexto desde los archivos fijos.
    """
    global df_portfolio_context, thesis_context_text
    print("--- Cargando datos de contexto fijos al iniciar la aplicación... ---")
    
    try:
        # Cargar el portafolio histórico
        df_portfolio_context = pd.read_csv(HISTORICAL_PORTFOLIO_PATH)
        print(f"-> Portafolio histórico '{HISTORICAL_PORTFOLIO_PATH}' cargado exitosamente. {len(df_portfolio_context)} registros encontrados.")
        
        # Cargar y extraer texto del PDF de contexto
        with open(CONTEXT_PDF_PATH, "rb") as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            thesis_context_text = "".join(page.get_text() for page in doc)
        print(f"-> PDF de contexto '{CONTEXT_PDF_PATH}' cargado y procesado exitosamente. {len(thesis_context_text)} caracteres leídos.")
        
    except FileNotFoundError as e:
        print(f"!!! ERROR CRÍTICO: No se encontró un archivo de contexto: {e}. La API podría no funcionar como se espera.")
    except Exception as e:
        print(f"!!! ERROR CRÍTICO al cargar datos de contexto: {e}.")

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

#def save_to_excel_incrementally(result_data: dict, output_filename: str):
#    try:
#        df_new_row = pd.json_normalize(result_data, sep='_') # Usamos '_' como separador
#        if not os.path.exists(output_filename):
#            df_new_row.to_excel(output_filename, index=False, engine='openpyxl')
#        else:
#            df_existing = pd.read_excel(output_filename, engine='openpyxl')
#            df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
#            df_combined.to_excel(output_filename, index=False, engine='openpyxl')
#        print(f"    -> Guardado exitoso en Excel.")
#    except Exception as e:
#        print(f"\n    !!! ADVERTENCIA: No se pudo guardar en Excel: {e} !!!")



# --- ENDPOINTS ---
async def run_scoring_loop(df_to_score, df_context, thesis_context): # Ya no necesita output_filename
    portfolio_context = analyze_portfolio(df_context)
    feedback_examples = get_feedback_examples(df_context)
    results = [] # Inicia una lista vacía
    total_startups = len(df_to_score)

    for index, row in df_to_score.iterrows():
        startup_name = row.get('Nombre de la startup') or row.get('Nombre', f'Fila {index + 1}')
        print(f"\n[ {index + 1} / {total_startups} ] Procesando: '{startup_name}'...")
        
        # La llamada a la IA se mantiene igual
        llm_result = get_llm_dimensional_scoring(row.to_json(), feedback_examples, thesis_context, portfolio_context)
        
        dimensional_scores = llm_result.get("dimensional_scores", {})
        final_score = sum((dimensional_scores.get(c, 0)) * d["peso"] for c, d in SCORING_CONFIG.items())
        
        original_data = row.where(pd.notna(row), None).to_dict()
        result_row = {**original_data, **llm_result, "final_weighted_score": round(final_score, 2)}
        
        # En lugar de guardar en Excel, añade el resultado a la lista
        results.append(result_row)
        
        print(f" -> Análisis completo para '{startup_name}'. Score: {result_row['final_weighted_score']}")
        time.sleep(1.1) # Mantenemos la pausa para no saturar la API
        
    return results

@app.post("/api/analyze-new-deals")
async def analyze_new_deals(new_deals_file: UploadFile = File(...)):
    """
    Este endpoint recibe un único archivo CSV/Excel con nuevas postulaciones,
    lo analiza usando el contexto histórico y el PDF ya cargados en memoria,
    y devuelve los resultados.
    """
    if df_portfolio_context is None or not thesis_context_text:
        raise HTTPException(
            status_code=503, 
            detail="Error del servidor: Los datos de contexto no se cargaron correctamente al iniciar. Revisa los logs del servidor."
        )
    
    try:
        content = await new_deals_file.read()
        df_to_score = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(...)

    print(f"\n--- INICIANDO SCORING PARA EL ARCHIVO '{new_deals_file.filename}' ---")
    
    # Llama a la lógica SIN el nombre de archivo de salida
    results = await run_scoring_loop(
        df_to_score=df_to_score,
        df_context=df_portfolio_context,
        thesis_context=thesis_context_text
    )
    
    print(f"\n--- PROCESO DE SCORING FINALIZADO PARA '{new_deals_file.filename}' ---")
    
    # Devuelve los resultados en formato JSON al frontend
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "../frontend"), html=True), name="frontend")
