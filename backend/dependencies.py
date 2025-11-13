# --- NUEVO ARCHIVO: backend/dependencies.py ---

import pandas as pd
from fastapi import HTTPException

# --- VARIABLES GLOBALES PARA CONTEXTO (CACHE) ---
# Movimos estas variables aquí desde main.py
df_portfolio_context: pd.DataFrame = None
thesis_context_text: str = ""

# --- DEPENDENCIAS PARA COMPARTIR CONTEXTO ---
# Movimos estas funciones aquí desde main.py
# Ahora, cualquier parte de la app puede importarlas desde aquí.
async def get_portfolio_context() -> pd.DataFrame:
    if df_portfolio_context is None:
        raise HTTPException(status_code=503, detail="El contexto del portafolio histórico no está cargado.")
    return df_portfolio_context

async def get_thesis_context() -> str:
    if not thesis_context_text:
        raise HTTPException(status_code=503, detail="El contexto de la tesis no está cargado.")
    return thesis_context_text