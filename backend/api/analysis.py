import io
import pandas as pd
import zipfile
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header, Body
from typing import Optional, Dict
from fastapi.responses import StreamingResponse
import traceback

# Importamos las funciones de scoring que necesitan los dos contextos
from services.scoring import run_scoring_loop_stream, run_single_scoring

# ¡CAMBIO CLAVE! Importamos las NUEVAS dependencias de contexto por separado
from dependencies import get_qualitative_context, get_quantitative_context, get_thesis_context

router = APIRouter()

@router.post("/api/analyze")
async def analyze_deals(
    new_deals_file: UploadFile = File(...),
    # ¡CAMBIO CLAVE! Usamos Depends para obtener cada contexto por separado
    df_qual_context: pd.DataFrame = Depends(get_qualitative_context),
    df_quant_context: pd.DataFrame = Depends(get_quantitative_context),
    thesis_context: str = Depends(get_thesis_context),
    # El header 'Accept' nos permite decidir si devolver un stream o no
    accept: Optional[str] = Header(None)
):
    """
    Endpoint inteligente para analizar un archivo de startups.
    - Si el cliente solicita 'text/event-stream', devuelve los resultados uno por uno.
    - De lo contrario (no implementado actualmente), devolvería un JSON completo.
    """
    print(f"\n--- RECIBIDA PETICIÓN DE ANÁLISIS PARA '{new_deals_file.filename}' ---")
    
    try:
        content = await new_deals_file.read()
        filename = new_deals_file.filename.lower()
        
        # DEBUG: Inspect content to diagnose upload issues
        print(f"DEBUG: Filename: {filename}")
        print(f"DEBUG: Content length: {len(content)}")
        
        # Check if content is a Data URL (starts with 'data:')
        # This happens if the frontend sends the file as a string instead of binary
        if content.startswith(b'data:'):
            print("⚠️ Data URL detected in backend. Decoding base64...")
            try:
                # Format is usually: data:application/vnd...;base64,UEsDB...
                header, encoded = content.split(b',', 1)
                content = base64.b64decode(encoded)
                print(f"✅ Decoded Data URL. New content length: {len(content)}")
                if len(content) > 20:
                    print(f"DEBUG: First 20 bytes after decode: {content[:20]}")
            except Exception as e:
                print(f"❌ Error decoding Data URL: {e}")
                # We continue with original content, though it will likely fail
        
        if len(content) > 20:
            print(f"DEBUG: First 20 bytes: {content[:20]}")
        else:
            print(f"DEBUG: Content bytes: {content}")
        
        if filename.endswith('.xlsx'):
            # Explicitly use openpyxl engine for .xlsx files
            try:
                df_to_score = pd.read_excel(io.BytesIO(content), engine='openpyxl')
            except zipfile.BadZipFile:
                print("⚠️ BadZipFile detected. The file might be a CSV renamed to .xlsx. Attempting to read as CSV.")
                df_to_score = pd.read_csv(io.BytesIO(content))
        else:
            # Asumimos CSV por defecto
            df_to_score = pd.read_csv(io.BytesIO(content))
        
        print(f"  -> Archivo leído. Shape: {df_to_score.shape}")
        print(f"  -> Columnas: {df_to_score.columns.tolist()}")
            
    except Exception as e:
        print(f"❌ ERROR CRÍTICO AL LEER EL ARCHIVO: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo (asegúrate de que sea CSV o Excel válido): {e}")

    # El frontend siempre pide un stream, así que esta es la ruta principal.
    if accept == "text/event-stream":
        print("--- INICIANDO ANÁLISIS EN MODO STREAMING ---")
        return StreamingResponse(
            run_scoring_loop_stream(df_to_score, df_qual_context, df_quant_context, thesis_context),
            media_type="text/event-stream"
        )
    else:
        # Esta parte no se está usando actualmente, pero la dejamos por si se necesita en el futuro.
        raise HTTPException(status_code=400, detail="Este endpoint solo soporta análisis en modo streaming. Asegúrate de incluir el header 'Accept: text/event-stream'.")

@router.post("/api/rerun-analysis")
async def rerun_single_analysis(
    startup_data: Dict = Body(...),
    # ¡CAMBIO CLAVE! También obtenemos los dos contextos para el re-análisis
    df_qual_context: pd.DataFrame = Depends(get_qualitative_context),
    df_quant_context: pd.DataFrame = Depends(get_quantitative_context),
    thesis_context: str = Depends(get_thesis_context)
):
    """
    Endpoint para re-analizar una única startup.
    Recibe los datos de la startup en formato JSON.
    """
    if not startup_data:
        raise HTTPException(status_code=400, detail="No se proporcionaron datos de la startup.")
    
    # ¡CAMBIO CLAVE! Pasamos los dos contextos a la función de re-análisis
    updated_startup = await run_single_scoring(
        startup_dict=startup_data,
        df_qual_context=df_qual_context,
        df_quant_context=df_quant_context,
        thesis_context=thesis_context
    )
    
    return updated_startup