import os
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

SERVICE_NAME = "ProcessTwin.Py"

STEPS_TOTAL = Counter(
    "processtwin_steps_total",
    "Total 5Hz PlantDigitalTwin simulation steps executed.",
    ["solver_mode"],
)
SP_REACTOR_TEMP_C = Gauge(
    "processtwin_sp_reactor_temp_c",
    "Series-Parallel Van de Vusse reactor temperature in Celsius.",
    ["reading"],
)
SP_CB_MOL_PER_L = Gauge(
    "processtwin_sp_cb_mol_per_l",
    "Series-Parallel desired product concentration [B] (mol/L).",
)
SP_EKF_NIS = Gauge(
    "processtwin_sp_ekf_nis",
    "Series-Parallel EKF6 chi-square Normalized Innovation Squared (NIS).",
)
SP_FAULT_ACTIVE = Gauge(
    "processtwin_sp_fault_active",
    "Whether chaos sensor fault injection is currently active (1=active, 0=none).",
)
HISTORIAN_OUTBOX_QUEUED = Gauge(
    "processtwin_historian_outbox_queued",
    "Pending Historian telemetry frames buffered in ProcessTwin SQLite outbox.",
)
HISTORIAN_REPLAYED_TOTAL = Gauge(
    "processtwin_historian_replayed_total",
    "Total Historian telemetry frames replayed from ProcessTwin outbox.",
)
HISTORIAN_FRAMES_SENT_TOTAL = Gauge(
    "processtwin_historian_frames_sent_total",
    "Total telemetry frames forwarded by ProcessTwin to Historian.",
)

_otel_configured = False
tracer = trace.get_tracer(SERVICE_NAME, "0.1.0")


def configure_observability(app: FastAPI) -> None:
    """Configure OpenTelemetry tracing, HTTPX propagation, X-Trace-Id header, and /metrics."""
    global _otel_configured, tracer

    if not _otel_configured:
        resource = Resource.create(
            {"service.name": SERVICE_NAME, "service.version": "0.1.0"}
        )
        provider = TracerProvider(resource=resource)

        otlp_traces = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        otlp_base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        target = otlp_traces or (
            f"{otlp_base.rstrip('/')}/v1/traces" if otlp_base else None
        )
        if target:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=target))
            )

        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(SERVICE_NAME, "0.1.0")
        HTTPXClientInstrumentor().instrument()
        _otel_configured = True

    FastAPIInstrumentor.instrument_app(app, excluded_urls="/metrics")

    @app.middleware("http")
    async def inject_trace_id_header(request: Request, call_next):
        response = await call_next(request)
        span_ctx = trace.get_current_span().get_span_context()
        if span_ctx.is_valid:
            response.headers["X-Trace-Id"] = format(span_ctx.trace_id, "032x")
        return response

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_step_metrics(frame: Any, bridge: Optional[Any] = None) -> None:
    """Update Prometheus counters/gauges from the latest twin telemetry frame."""
    sp = getattr(frame, "sp_control", None)
    solver_mode = getattr(sp, "solver_mode", "rk4") if sp else "rk4"
    STEPS_TOTAL.labels(solver_mode=solver_mode).inc()

    if sp is not None:
        SP_REACTOR_TEMP_C.labels(reading="true").set(float(sp.reactor_temp_c))
        measured = (
            sp.measured_reactor_temp_c
            if sp.measured_reactor_temp_c is not None
            else sp.reactor_temp_c
        )
        SP_REACTOR_TEMP_C.labels(reading="measured").set(float(measured))
        SP_CB_MOL_PER_L.set(float(sp.cb_mol_per_l))
        SP_EKF_NIS.set(float(sp.ekf_nis))
        SP_FAULT_ACTIVE.set(1.0 if sp.fault_active else 0.0)

    if bridge is not None:
        HISTORIAN_OUTBOX_QUEUED.set(float(getattr(bridge, "queued_frames", 0)))
        HISTORIAN_REPLAYED_TOTAL.set(float(getattr(bridge, "replayed_frames", 0)))
        HISTORIAN_FRAMES_SENT_TOTAL.set(float(getattr(bridge, "frames_sent", 0)))
