# --- CÓDIGO ACTUALIZADO: backend/main.py ---

import os
import pandas as pd
import fitz
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- CAMBIO CLAVE: Importamos desde los nuevos módulos ---
from api.analysis import router as analysis_router
# Importamos las variables de contexto para poder modificarlas
from dependencies import df_portfolio_context, thesis_context_text

# ... (El resto de la configuración de la app sigue igual)
load_dotenv()
app = FastAPI(
    title="Deal Flow AI API v2",
    description="API para puntuar postulaciones de startups y analizar datos históricos."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- RUTAS A ARCHIVOS DE CONTEXTO ---
HISTORICAL_PORTFOLIO_PATH = "historical_data.csv"
CONTEXT_PDF_PATH = "proyecto_preprofesional.pdf"

# --- EVENTO DE INICIO (STARTUP) ---
@app.on_event("startup")
async def startup_event():
    # Usamos 'global' para modificar las variables importadas desde dependencies.py
    global df_portfolio_context, thesis_context_text
    
    print("--- 🚀 Iniciando la aplicación y cargando datos de contexto... ---")
    
    # 1. Configurar API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❗️ ERROR CRÍTICO: GOOGLE_API_KEY no encontrada.")
    else:
        genai.configure(api_key=api_key)
        print(f"✅ API Key de Google cargada: {api_key[:4]}...")

    # 2. Cargar datos de contexto
    try:
        # Aquí asignamos los datos a las variables globales importadas
        df_portfolio_context = pd.read_csv(HISTORICAL_PORTFOLIO_PATH)
        print(f"✅ Portafolio histórico cargado. {len(df_portfolio_context)} registros.")
        
        with open(CONTEXT_PDF_PATH, "rb") as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            thesis_context_text = "".join(page.get_text() for page in doc)
        print(f"✅ PDF de contexto cargado. {len(thesis_context_text)} caracteres.")
        
        print("--- ✅ Carga de contexto finalizada. La API está lista. ---")

    except FileNotFoundError as e:
        print(f"❗️ ERROR CRÍTICO al cargar contexto: {e}.")
    except Exception as e:
        print(f"❗️ ERROR CRÍTICO al cargar datos: {e}.")


app.include_router(analysis_router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")