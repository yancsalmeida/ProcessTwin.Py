import os
from enum import IntEnum
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class IndustrialRole(IntEnum):
    ANONYMOUS = 0
    VIEWER = 1
    OPERATOR = 2
    ENGINEER = 3


def _extract_credential(request: Request) -> Optional[str]:
    api_key = request.headers.get("X-Api-Key")
    if api_key and api_key.strip():
        return api_key.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return None


def _required_role_for(method: str, path: str) -> IndustrialRole:
    if method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
        return IndustrialRole.ANONYMOUS

    if (
        path.startswith("/api/faults/")
        or path.startswith("/api/series-parallel/control/config")
        or path.startswith("/api/series-parallel/control/reset")
        or path.startswith("/api/interlock/reset")
    ):
        return IndustrialRole.ENGINEER

    if (
        path.startswith("/api/actuators")
        or path.startswith("/api/nmpc/")
        or path.startswith("/api/bridge/flush-outbox")
        or path.startswith("/api/alarms/")
    ):
        return IndustrialRole.OPERATOR

    return IndustrialRole.ANONYMOUS


def configure_rbac(app: FastAPI) -> None:
    """Attach lightweight X-Api-Key / Bearer RBAC guardrails on critical mutating endpoints."""

    @app.middleware("http")
    async def industrial_rbac_middleware(request: Request, call_next):
        enforce = os.getenv("INDUSTRIAL_ENFORCE_RBAC", "false").lower() == "true"
        viewer_key = os.getenv("INDUSTRIAL_VIEWER_KEY", "viewer-ro-key")
        operator_key = os.getenv("INDUSTRIAL_OPERATOR_KEY", "operator-ctrl-key")
        engineer_key = os.getenv("INDUSTRIAL_ENGINEER_KEY", "engineer-admin-key")

        required = _required_role_for(request.method, request.url.path)
        cred = _extract_credential(request)

        actor = IndustrialRole.ANONYMOUS
        if cred is not None:
            if cred == engineer_key:
                actor = IndustrialRole.ENGINEER
            elif cred == operator_key:
                actor = IndustrialRole.OPERATOR
            elif cred == viewer_key:
                actor = IndustrialRole.VIEWER
            else:
                return JSONResponse(
                    status_code=401,
                    content={
                        "title": "Invalid API Key",
                        "status": 401,
                        "detail": "The supplied X-Api-Key or Bearer token is not recognized.",
                    },
                )

        if required > IndustrialRole.ANONYMOUS:
            if actor == IndustrialRole.ANONYMOUS and enforce:
                return JSONResponse(
                    status_code=401,
                    content={
                        "title": "Authentication Required",
                        "status": 401,
                        "detail": f"Missing X-Api-Key for protected route requiring role '{required.name}'.",
                    },
                )

            if actor != IndustrialRole.ANONYMOUS and actor < required:
                return JSONResponse(
                    status_code=403,
                    content={
                        "title": "Insufficient Role",
                        "status": 403,
                        "detail": f"Role '{actor.name}' cannot execute {request.method} {request.url.path}; requires '{required.name}'.",
                    },
                )

        response = await call_next(request)
        if actor > IndustrialRole.ANONYMOUS:
            response.headers["X-Auth-Role"] = actor.name
        return response
