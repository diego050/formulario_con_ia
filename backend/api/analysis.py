import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header, Body
from typing import Optional, Dict
from fastapi.responses import StreamingResponse

# Importamos TODAS las funciones de scoring que necesitamos
from services.scoring import run_scoring_loop, run_scoring_loop_stream, run_single_scoring
from dependencies import get_portfolio_context, get_thesis_context

router = APIRouter()

@router.post("/api/analyze")
async def analyze_deals(
    new_deals_file: UploadFile = File(...),
    df_portfolio_context: pd.DataFrame = Depends(get_portfolio_context),
    thesis_context_text: str = Depends(get_thesis_context),
    accept: Optional[str] = Header(None) # Header para detectar si se pide un stream
):
    """
    Endpoint inteligente:
    - Si el header 'Accept' es 'text/event-stream', devuelve un stream.
    - De lo contrario, devuelve un JSON completo al final.
    """
    print(f"\n--- RECIBIDA PETICIÓN DE ANÁLISIS PARA '{new_deals_file.filename}' ---")
    
    try:
        content = await new_deals_file.read()
        df_to_score = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Error al leer el archivo CSV.")

    if accept == "text/event-stream":
        print("--- INICIANDO ANÁLISIS EN MODO STREAMING ---")
        return StreamingResponse(
            run_scoring_loop_stream(df_to_score, df_portfolio_context, thesis_context_text),
            media_type="text/event-stream"
        )
    else:
        print("--- INICIANDO ANÁLISIS EN MODO BATCH (JSON) ---")
        results = await run_scoring_loop(df_to_score, df_portfolio_context, thesis_context_text)
        print(f"--- ANÁLISIS BATCH FINALIZADO ---")
        return results

@router.post("/api/rerun-analysis")
async def rerun_single_analysis(
    startup_data: Dict = Body(...),
    df_portfolio_context: pd.DataFrame = Depends(get_portfolio_context),
    thesis_context_text: str = Depends(get_thesis_context)
):
    if not startup_data:
        raise HTTPException(status_code=400, detail="No se proporcionaron datos de la startup.")
    
    updated_startup = await run_single_scoring(
        startup_dict=startup_data,
        df_context=df_portfolio_context,
        thesis_context=thesis_context_text
    )
    return updated_startup