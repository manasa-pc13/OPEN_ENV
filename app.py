import logging
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import Action, DataCleaningEnv, TASKS

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Cleaning Agent Environment API")

# Global environment instance (in-memory for simple Space evaluation)
global_env = DataCleaningEnv(task_id="easy")


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


@app.get("/")
def health_check():
    """Automated ping endpoint for Hugging Face Spaces"""
    return JSONResponse(status_code=200, content={"status": "ok", "message": "Environment is up!"})


@app.post("/reset")
def reset_env(req: ResetRequest = Body(default=ResetRequest())):
    """Reset the environment state and return initial observation."""
    global global_env
    task_id = req.task_id if req.task_id else "easy"
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id='{task_id}'. Valid: {list(TASKS.keys())}")
    
    global_env = DataCleaningEnv(task_id=task_id)
    obs = global_env.reset()
    return obs.model_dump()


@app.post("/step")
def step_env(action: Action = Body(...)):
    """Take a step in the environment by applying an action."""
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }


@app.get("/state")
def state_env():
    """Return the entire internal state (for debugging and testing)."""
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return global_env.state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
