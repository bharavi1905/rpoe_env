# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Rotary Parking Optimization Environment (RPOE).

This module creates an HTTP server that exposes RotaryParkingEnv
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /tasks: List all available evaluation tasks
    - POST /task/{task_id}: Run a named task with a random agent

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m rpoe_env.server.app
"""

import random

import uvicorn
from fastapi import HTTPException

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from models import RPOEAction, RPOEObservation, ActionType, TaskResult
from server.env import RotaryParkingEnv
from tasks.graders import TASKS


# Create the app with web interface and README integration
app = create_app(
    RotaryParkingEnv,
    RPOEAction,
    RPOEObservation,
    env_name="rpoe",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Additional endpoints
# ---------------------------------------------------------------------------

_TASK_META = [
    {
        "id":          "task1_easy",
        "difficulty":  "easy",
        "steps":       50,
        "description": "Optimal rotation planning — low arrival rate (λ=0.1), no overflow risk.",
    },
    {
        "id":          "task2_medium",
        "difficulty":  "medium",
        "steps":       180,
        "description": "Peak-hour throughput — morning peak (λ=0.35), score = served / (served + overflowed).",
    },
    {
        "id":          "task3_hard",
        "difficulty":  "hard",
        "steps":       1080,
        "description": "Full 18-hour day — composite score: throughput (40%), efficiency (25%), retrieval (20%), stability (15%).",
    },
]


@app.get("/tasks")
def list_tasks():
    """Return metadata for all three evaluation tasks."""
    return {"tasks": _TASK_META}


@app.post("/task/{task_id}")
def run_task(task_id: str, seed: int = 42) -> TaskResult:
    """
    Run the named task with a random agent and return a TaskResult.

    Args:
        task_id: One of task1_easy, task2_medium, task3_hard.
        seed:    RNG seed for reproducibility (default 42).
    """
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task '{task_id}'. Valid options: {list(TASKS.keys())}",
        )

    def random_agent(_):
        return RPOEAction(action=random.choice(list(ActionType)))

    return TASKS[task_id](random_agent, seed=seed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 7860
        python -m rpoe_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn rpoe_env.server.app:app --workers 4
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
