"""
FastAPI inference server for SafeSched deadlock prediction.

Input schema follows matrix-based system state and is transformed into the
same feature order used by the trained Logistic Regression pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "logistic_regression_deadlock.joblib"
DEFAULT_THRESHOLD = 0.02


class PredictRequest(BaseModel):
    processes: int
    resources: int
    allocation_matrix: List[List[int]]
    max_matrix: List[List[int]]
    available: List[int]


class PredictResponse(BaseModel):
    deadlock: int
    confidence: float
    # --- NEW FIELDS FOR PROBLEM #38 COMPLIANCE ---
    logical_deadlock: int
    deadlocked_processes: List[int]
    safe_sequence: List[int]
    immediate_runnable: int
    recommended_action: str = ""


app = FastAPI(title="SafeSched Deadlock Prediction API", version="1.0.0")

MODEL_PIPELINE: Any = None
FEATURE_COLUMNS: List[str] = []
PREDICTION_THRESHOLD: float = DEFAULT_THRESHOLD


def compute_deadlock_from_state(payload: PredictRequest) -> Dict[str, Any]:
    """
    Determine deadlock using a matrix safety-progress check.

    A process can complete if NEED <= current work vector.
    Processes holding no resources are treated as not participating in deadlock.
    """
    p = payload.processes
    r = payload.resources

    allocation = [row[:] for row in payload.allocation_matrix]
    maximum = [row[:] for row in payload.max_matrix]
    work = payload.available[:]

    need = [
        [maximum[i][j] - allocation[i][j] for j in range(r)]
        for i in range(p)
    ]

    # Calculate runnability strictly: can they run AND do they hold anything?
    runnable_indices = [
        i for i in range(p)
        if all(need[i][j] <= (work[j] if j < len(work) else 0) for j in range(r))
    ]
    
    # Contributory means they can run and will release resources back to the pool
    contributory_runnable = [
        i for i in runnable_indices
        if any(allocation[i][j] > 0 for j in range(r))
    ]

    # Initial state: processes that hold nothing AND need nothing are effectively finished
    finish = [
        all(allocation[i][j] == 0 for j in range(r)) and all(need[i][j] <= 0 for j in range(r))
        for i in range(p)
    ]
    
    # Process safety algorithm
    current_work = payload.available[:]  # Use a local copy to avoid modifying original available
    safe_sequence = []
    
    # Banker's safety check
    while True:
        found_next = False
        for i in range(p):
            if not finish[i]:
                # Can process i be satisfied?
                if all(need[i][j] <= current_work[j] for j in range(r)):
                    # Release held resources to work vector
                    for j in range(r):
                        current_work[j] += allocation[i][j]
                    finish[i] = True
                    safe_sequence.append(i)
                    found_next = True
                    # Restart once we've made progress to re-check all blocked processes
        if not found_next:
            break

    # If any process that holds resources is not finished, we have a deadlock
    deadlocked_processes = [i for i in range(p) if not finish[i] and any(allocation[i][j] > 0 for j in range(r))]
    logical_deadlock = 1 if len(deadlocked_processes) > 0 else 0

    # Simple resolution strategy: if deadlocked, suggest process with most resources
    recommended_action = ""
    if logical_deadlock:
        resource_counts = [sum(allocation[pid]) for pid in deadlocked_processes]
        victim_idx = resource_counts.index(max(resource_counts))
        victim_pid = deadlocked_processes[victim_idx]
        recommended_action = f"Terminate Process {victim_pid} to release {sum(allocation[victim_pid])} resources."

    return {
        "logical_deadlock": logical_deadlock,
        "deadlocked_processes": deadlocked_processes,
        "safe_sequence": safe_sequence,
        "immediate_runnable": len(runnable_indices),
        "contributory_feasible_now": len(contributory_runnable),
        "recommended_action": recommended_action,
    }


def load_model_artifact(path: Path) -> Tuple[Any, List[str], float]:
    """Load trained artifact once at startup."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)

    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        feature_columns = list(artifact.get("feature_columns", []))
        threshold = float(artifact.get("decision_threshold", DEFAULT_THRESHOLD))
        return pipeline, feature_columns, threshold

    if hasattr(artifact, "predict"):
        return artifact, [], DEFAULT_THRESHOLD

    raise ValueError("Unsupported model artifact format.")


def validate_state(payload: PredictRequest) -> None:
    """Validate matrix sizes and value constraints with clear messages."""
    p = payload.processes
    r = payload.resources

    if p <= 0:
        raise HTTPException(status_code=400, detail="'processes' must be > 0.")
    if r <= 0:
        raise HTTPException(status_code=400, detail="'resources' must be > 0.")

    if len(payload.allocation_matrix) != p:
        raise HTTPException(
            status_code=400,
            detail=f"allocation_matrix row count must equal processes ({p}).",
        )
    if len(payload.max_matrix) != p:
        raise HTTPException(
            status_code=400,
            detail=f"max_matrix row count must equal processes ({p}).",
        )
    if len(payload.available) != r:
        raise HTTPException(
            status_code=400,
            detail=f"available length must equal resources ({r}).",
        )

    for i, row in enumerate(payload.allocation_matrix):
        if len(row) != r:
            raise HTTPException(
                status_code=400,
                detail=f"allocation_matrix row {i} must have {r} columns.",
            )
    for i, row in enumerate(payload.max_matrix):
        if len(row) != r:
            raise HTTPException(
                status_code=400,
                detail=f"max_matrix row {i} must have {r} columns.",
            )

    for i in range(p):
        for j in range(r):
            alloc = payload.allocation_matrix[i][j]
            mx = payload.max_matrix[i][j]
            if alloc < 0 or mx < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Negative values are not allowed (process={i}, resource={j}).",
                )
            if alloc > mx:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"allocation cannot exceed max at process={i}, resource={j} "
                        f"(allocation={alloc}, max={mx})."
                    ),
                )

    if any(v < 0 for v in payload.available):
        raise HTTPException(status_code=400, detail="'available' cannot contain negative values.")


def build_engineered_features(payload: PredictRequest) -> Dict[str, float]:
    """Create derived system-state features from matrices."""
    p = payload.processes
    r = payload.resources

    total_allocation = float(sum(sum(row) for row in payload.allocation_matrix))
    total_max = float(sum(sum(row) for row in payload.max_matrix))
    total_need = float(max(total_max - total_allocation, 0.0))
    total_available = float(sum(payload.available))

    num_executable_processes = int(
        sum(
            1
            for i in range(p)
            if all(
                payload.max_matrix[i][j] - payload.allocation_matrix[i][j] <= payload.available[j]
                for j in range(r)
            )
        )
    )
    blocked_processes = int(max(p - num_executable_processes, 0))

    # Count how many can run and WILL release resources (the 'contributory' check)
    contributory_feasible_now = 0
    for i in range(p):
        can_run = all(
            payload.max_matrix[i][j] - payload.allocation_matrix[i][j] <= payload.available[j]
            for j in range(r)
        )
        has_resources = any(payload.allocation_matrix[i][j] > 0 for j in range(r))
        if can_run and has_resources:
            contributory_feasible_now += 1

    total_resources = float(total_allocation + total_available)
    fraction_blocked = float(blocked_processes / p) if p > 0 else 0.0
    utilization_ratio = float(total_allocation / total_resources) if total_resources > 0 else 0.0
    need_to_available_ratio = (
        float(total_need / total_available) if total_available > 0 else float(total_need)
    )

    return {
        "num_processes": float(p),
        "num_resources": float(r),
        "distributed": 0.0,
        "dynamic_join_leave": 0.0,
        "total_allocation": total_allocation,
        "total_max": total_max,
        "total_need": total_need,
        "total_available": total_available,
        "num_executable_processes": float(num_executable_processes),
        "contributory_feasible_now": float(contributory_feasible_now), # New Feature for the model
        "blocked_processes": float(blocked_processes),
        "fraction_blocked": fraction_blocked,
        "utilization_ratio": utilization_ratio,
        "need_to_available_ratio": need_to_available_ratio,
        # Backward-compatible aliases if older artifacts expect these names.
        "total_resources": total_resources,
        "sum_available": total_available,
        "unmet_need_processes": float(blocked_processes),
        "feasible_processes_now": float(num_executable_processes),
        "unmet_demand_vs_supply": float(max(total_need - total_available, 0.0)),
        "resource_utilization_ratio": utilization_ratio,
    }


def to_model_frame(engineered: Dict[str, float]) -> pd.DataFrame:
    """Project engineered features into exact training feature order."""
    if FEATURE_COLUMNS:
        row = {col: engineered.get(col, 0.0) for col in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return pd.DataFrame([engineered])


@app.on_event("startup")
def startup_load_model() -> None:
    """Load model once for all requests."""
    global MODEL_PIPELINE, FEATURE_COLUMNS, PREDICTION_THRESHOLD
    MODEL_PIPELINE, FEATURE_COLUMNS, PREDICTION_THRESHOLD = load_model_artifact(MODEL_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PIPELINE is not None,
        "model_path": str(MODEL_PATH),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": PREDICTION_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict deadlock from matrix state payload."""
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    validate_state(payload)
    engineered = build_engineered_features(payload)
    model_input = to_model_frame(engineered)
    
    # Get logical/Banker's info from Simulation logic
    logic_results = compute_deadlock_from_state(payload)

    try:
        if hasattr(MODEL_PIPELINE, "predict_proba"):
            deadlock_probability = float(MODEL_PIPELINE.predict_proba(model_input)[0][1])
            deadlock = int(deadlock_probability >= PREDICTION_THRESHOLD)
            confidence = deadlock_probability if deadlock == 1 else (1.0 - deadlock_probability)
        elif hasattr(MODEL_PIPELINE, "predict"):
            deadlock = int(MODEL_PIPELINE.predict(model_input)[0])
            confidence = float(deadlock)
        else:
            raise HTTPException(status_code=500, detail="Loaded model does not support prediction.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        deadlock=deadlock,
        confidence=confidence,
        logical_deadlock=logic_results["logical_deadlock"],
        deadlocked_processes=logic_results["deadlocked_processes"],
        safe_sequence=logic_results["safe_sequence"],
        immediate_runnable=logic_results["immediate_runnable"],
        recommended_action=logic_results["recommended_action"]
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
