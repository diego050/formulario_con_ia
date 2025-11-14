import pandas as pd
from fastapi import HTTPException

# Creamos un diccionario para mantener el estado de la aplicación.
# Esta es una forma más robusta de compartir datos que las variables globales.
app_state = {
    "df_portfolio_context": None,
    "thesis_context_text": ""
}

# --- DEPENDENCIAS PARA COMPARTIR CONTEXTO ---
# Ahora leen los datos desde el diccionario 'app_state'.
async def get_portfolio_context() -> pd.DataFrame:
    if app_state["df_portfolio_context"] is None:
        raise HTTPException(status_code=503, detail="El contexto del portafolio histórico no está cargado.")
    return app_state["df_portfolio_context"]

async def get_thesis_context() -> str:
    if not app_state["thesis_context_text"]:
        raise HTTPException(status_code=503, detail="El contexto de la tesis no está cargado.")
    return app_state["thesis_context_text"]