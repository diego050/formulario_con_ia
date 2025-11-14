from fastapi import APIRouter, HTTPException
import json

router = APIRouter()

@router.get("/api/config/scoring-weights")
async def get_scoring_weights():
    """
    Endpoint para servir la configuración de los pesos de scoring
    desde el archivo scoring_config.json.
    """
    try:
        with open("scoring_config.json", "r") as f:
            config_data = json.load(f)
        
        # Devolvemos solo la parte que le interesa al frontend
        scoring_categories = config_data.get("SCORING_CATEGORIES")
        if scoring_categories is None:
            raise HTTPException(status_code=404, detail="La clave 'SCORING_CATEGORIES' no se encontró en el archivo de configuración.")
        
        return scoring_categories
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="El archivo 'scoring_config.json' no se encontró en el servidor.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo de configuración: {e}")