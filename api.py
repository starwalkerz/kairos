"""
api.py — async FastAPI for KAIROS.

Endpoints:
    GET  /                          browser UI
    GET  /api/health                healthcheck
    GET  /api/cycles/presets        synodic period presets
    POST /api/cycles/evaluate       baseline for a time grid and cycle stack
    POST /api/detect                full detection run
    POST /api/adapters/register     register an external adapter
    POST /api/adapters/{name}/call  dispatch a payload to an adapter
    WS   /ws/detect                 live detection stream
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from core import GmGCycle, SYNODIC_PRESETS, baseline
from detect import detect
from adapters import registry as adapter_registry


EXECUTOR = ProcessPoolExecutor(max_workers=4)


async def _run_detect(**kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, lambda: detect(**kwargs))


class CycleSpec(BaseModel):
    name: str = "custom"
    period: float
    phase0: float = 0.0
    mu_L: float
    sigma_L: float = Field(gt=0)
    mu_T: float
    sigma_T: float = Field(gt=0)
    amplitude: float = 1.0

    def to_cycle(self) -> GmGCycle:
        return GmGCycle(**self.model_dump())


class DetectRequest(BaseModel):
    series: list[float]
    t: list[float]
    cycles: list[CycleSpec]
    kappa: float = Field(default=2.0, gt=0)
    threshold: float = Field(default=0.05, gt=0, lt=1)
    qae_eval_qubits: int = Field(default=6, ge=2, le=20)
    quantum: bool = True
    mark_threshold: Optional[float] = None

    @field_validator("series", "t")
    @classmethod
    def _nonempty(cls, v):
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("t")
    @classmethod
    def _match_length(cls, v, info):
        if "series" in info.data and len(v) != len(info.data["series"]):
            raise ValueError("series and t must have equal length")
        return v


class EvaluateRequest(BaseModel):
    t: list[float]
    cycles: list[CycleSpec]


class AdapterRegister(BaseModel):
    name: str
    base_url: str
    path: str


class AdapterCall(BaseModel):
    payload: dict


app = FastAPI(title="KAIROS", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_FILE = Path(__file__).parent / "ui.html"


@app.get("/", include_in_schema=False)
async def root():
    if UI_FILE.exists():
        return FileResponse(UI_FILE, media_type="text/html")
    return {"message": "KAIROS API running. See /docs."}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/cycles/presets")
async def presets():
    return SYNODIC_PRESETS


@app.post("/api/cycles/evaluate")
async def evaluate_cycles(req: EvaluateRequest):
    try:
        tt = np.asarray(req.t, dtype=float)
        cc = [c.to_cycle() for c in req.cycles]
        return {"baseline": baseline(tt, cc).tolist()}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/detect")
async def api_detect(req: DetectRequest):
    try:
        cycles = [c.to_cycle() for c in req.cycles]
        report = await _run_detect(
            series=req.series,
            t=req.t,
            cycles=cycles,
            kappa=req.kappa,
            threshold=req.threshold,
            qae_eval_qubits=req.qae_eval_qubits,
            quantum=req.quantum,
            mark_threshold=req.mark_threshold,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return report.to_handoff()


@app.post("/api/adapters/register")
async def register_adapter(req: AdapterRegister):
    try:
        adapter_registry.register(req.name, req.base_url, req.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"registered": adapter_registry.names()}


@app.post("/api/adapters/{name}/call")
async def call_adapter(name: str, req: AdapterCall):
    try:
        return adapter_registry.call(name, req.payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{type(exc).__name__}: {exc}",
        )


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") != "detect":
                continue
            try:
                cycles = [CycleSpec(**c).to_cycle() for c in msg["cycles"]]
                report = await _run_detect(
                    series=msg["series"],
                    t=msg["t"],
                    cycles=cycles,
                    kappa=msg.get("kappa", 2.0),
                    threshold=msg.get("threshold", 0.05),
                    quantum=msg.get("quantum", True),
                )
                await ws.send_json(report.to_handoff())
            except Exception as exc:
                await ws.send_json(
                    {"error": f"{type(exc).__name__}: {exc}"}
                )
    except WebSocketDisconnect:
        pass