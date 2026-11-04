"""
Native C-ABI FFI Bridge connecting ProcessTwin.Py with the high-performance ReactorCore.C numerical engine.

Loads libreactorcore.so dynamically via ctypes, providing:
1. Low-level struct marshalling (CstrParamsFFI, CstrStateFFI).
2. O(1) single-step RK4 integration via thread-local pre-allocated workspaces in C.
3. Batch trajectory simulation running entirely in native C11 (zero Python interpreter loop overhead).
4. Pure Python reference fallback implementation for automated parity validation and cross-benchmarking.
"""

from __future__ import annotations
import ctypes
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


class CstrParamsFFI(ctypes.Structure):
    """
    C-ABI compatible layout matching CstrParamsFFI in ReactorCore.C/include/reactor_ffi.h.
    Total size: 80 bytes (10 x 64-bit IEEE-754 doubles).
    """
    _fields_ = [
        ("volumetric_flow_l_per_min", ctypes.c_double),
        ("volume_liters", ctypes.c_double),
        ("k0_per_min", ctypes.c_double),
        ("ea_over_r_kelvin", ctypes.c_double),
        ("delta_h_j_per_mol", ctypes.c_double),
        ("rho_cp_j_per_l_k", ctypes.c_double),
        ("ua_j_per_min_k", ctypes.c_double),
        ("ca_feed_mol_per_l", ctypes.c_double),
        ("t_feed_kelvin", ctypes.c_double),
        ("t_coolant_kelvin", ctypes.c_double),
    ]


class CstrStateFFI(ctypes.Structure):
    """
    C-ABI compatible layout matching CstrStateFFI in ReactorCore.C/include/reactor_ffi.h.
    """
    _fields_ = [
        ("ca_mol_per_l", ctypes.c_double),
        ("temp_kelvin", ctypes.c_double),
        ("conversion_pct", ctypes.c_double),
        ("reaction_rate_mol_per_l_min", ctypes.c_double),
        ("is_runaway", ctypes.c_bool),
    ]


@dataclass
class NativeCstrParams:
    volumetric_flow_l_per_min: float = 100.0
    volume_liters: float = 100.0
    k0_per_min: float = 7.08e10
    ea_over_r_kelvin: float = 8330.1
    delta_h_j_per_mol: float = 200000.0
    rho_cp_j_per_l_k: float = 4000.0
    ua_j_per_min_k: float = 40000.0
    ca_feed_mol_per_l: float = 1.0
    t_feed_kelvin: float = 350.0
    t_coolant_kelvin: float = 300.0

    def to_ffi(self) -> CstrParamsFFI:
        return CstrParamsFFI(
            volumetric_flow_l_per_min=self.volumetric_flow_l_per_min,
            volume_liters=self.volume_liters,
            k0_per_min=self.k0_per_min,
            ea_over_r_kelvin=self.ea_over_r_kelvin,
            delta_h_j_per_mol=self.delta_h_j_per_mol,
            rho_cp_j_per_l_k=self.rho_cp_j_per_l_k,
            ua_j_per_min_k=self.ua_j_per_min_k,
            ca_feed_mol_per_l=self.ca_feed_mol_per_l,
            t_feed_kelvin=self.t_feed_kelvin,
            t_coolant_kelvin=self.t_coolant_kelvin,
        )

    @classmethod
    def from_ffi(cls, ffi: CstrParamsFFI) -> NativeCstrParams:
        return cls(
            volumetric_flow_l_per_min=ffi.volumetric_flow_l_per_min,
            volume_liters=ffi.volume_liters,
            k0_per_min=ffi.k0_per_min,
            ea_over_r_kelvin=ffi.ea_over_r_kelvin,
            delta_h_j_per_mol=ffi.delta_h_j_per_mol,
            rho_cp_j_per_l_k=ffi.rho_cp_j_per_l_k,
            ua_j_per_min_k=ffi.ua_j_per_min_k,
            ca_feed_mol_per_l=ffi.ca_feed_mol_per_l,
            t_feed_kelvin=ffi.t_feed_kelvin,
            t_coolant_kelvin=ffi.t_coolant_kelvin,
        )


@dataclass
class NativeCstrState:
    ca_mol_per_l: float
    temp_kelvin: float
    conversion_pct: float
    reaction_rate_mol_per_l_min: float
    is_runaway: bool


@dataclass
class TrajectoryResult:
    times_min: List[float]
    ca_mol_per_l: List[float]
    temp_kelvin: List[float]


def find_reactor_core_library() -> Optional[str]:
    """
    Searches for libreactorcore.so in standard project locations and environment paths.
    """
    env_path = os.environ.get("REACTOR_CORE_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check relative to this module in ProcessTwin.Py
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Local to current repo
        os.path.join(module_dir, "libreactorcore.so"),
        os.path.join(module_dir, "../../../libreactorcore.so"),
        # Relative to workspace: ../ReactorCore.C/libreactorcore.so
        os.path.abspath(os.path.join(module_dir, "../../../../../ReactorCore.C/libreactorcore.so")),
        # Absolute system paths
        "/home/uuu/Documents/Projetos/ReactorCore.C/libreactorcore.so",
        "/usr/local/lib/libreactorcore.so",
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


class NativeReactorCore:
    """
    High-level Python wrapper around the compiled ReactorCore.C shared object library.
    Falls back gracefully to a pure Python reference solver if the binary is absent.
    """

    def __init__(self, lib_path: Optional[str] = None) -> None:
        self.lib_path = lib_path or find_reactor_core_library()
        self._lib: Optional[ctypes.CDLL] = None
        self._load_library()

    def _load_library(self) -> None:
        if not self.lib_path or not os.path.isfile(self.lib_path):
            return

        try:
            lib = ctypes.CDLL(self.lib_path)

            # Bind: const char* reactor_core_version(void)
            lib.reactor_core_version.argtypes = []
            lib.reactor_core_version.restype = ctypes.c_char_p

            # Bind: int reactor_cstr_default_params(CstrParamsFFI *out_params)
            lib.reactor_cstr_default_params.argtypes = [ctypes.POINTER(CstrParamsFFI)]
            lib.reactor_cstr_default_params.restype = ctypes.c_int

            # Bind: double reactor_cstr_reaction_rate(const CstrParamsFFI *params, double Ca, double T)
            lib.reactor_cstr_reaction_rate.argtypes = [ctypes.POINTER(CstrParamsFFI), ctypes.c_double, ctypes.c_double]
            lib.reactor_cstr_reaction_rate.restype = ctypes.c_double

            # Bind: int reactor_cstr_step_rk4(const CstrParamsFFI *params, double *ca, double *temp_k, double dt)
            lib.reactor_cstr_step_rk4.argtypes = [
                ctypes.POINTER(CstrParamsFFI),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
            ]
            lib.reactor_cstr_step_rk4.restype = ctypes.c_int

            # Bind: int reactor_cstr_step_detailed(...)
            lib.reactor_cstr_step_detailed.argtypes = [
                ctypes.POINTER(CstrParamsFFI),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(CstrStateFFI),
            ]
            lib.reactor_cstr_step_detailed.restype = ctypes.c_int

            # Bind: int reactor_cstr_simulate_trajectory(...)
            lib.reactor_cstr_simulate_trajectory.argtypes = [
                ctypes.POINTER(CstrParamsFFI),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.reactor_cstr_simulate_trajectory.restype = ctypes.c_int

            self._lib = lib
        except Exception:
            self._lib = None

    @property
    def is_available(self) -> bool:
        """True if the native C shared library is loaded and operational."""
        return self._lib is not None

    def get_version(self) -> str:
        """Returns the version and architecture banner reported by the C library."""
        if not self._lib:
            return "Python-Reference-Fallback (No Native .so Loaded)"
        raw = self._lib.reactor_core_version()
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def get_default_params(self) -> NativeCstrParams:
        """Retrieves standard benchmark literature physical parameters."""
        if not self._lib:
            return NativeCstrParams()
        ffi_params = CstrParamsFFI()
        rc = self._lib.reactor_cstr_default_params(ctypes.byref(ffi_params))
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_default_params failed with code {rc}")
        return NativeCstrParams.from_ffi(ffi_params)

    def reaction_rate(self, params: NativeCstrParams, ca: float, temp_k: float) -> float:
        """Calculates instantaneous Arrhenius reaction rate: r = k0 * exp(-Ea / (R*T)) * Ca."""
        if not self._lib:
            return self._py_reaction_rate(params, ca, temp_k)
        ffi_p = params.to_ffi()
        return float(self._lib.reactor_cstr_reaction_rate(ctypes.byref(ffi_p), ca, temp_k))

    def step_rk4(
        self,
        params: NativeCstrParams,
        ca: float,
        temp_k: float,
        dt_min: float = 0.005,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Advances the state by dt_min using 4th-Order Runge-Kutta in C.
        Returns: (new_ca, new_temp_k).
        """
        if dt is not None:
            dt_min = dt

        if not self._lib:
            return self._py_step_rk4(params, ca, temp_k, dt_min)

        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_temp = ctypes.c_double(temp_k)
        rc = self._lib.reactor_cstr_step_rk4(
            ctypes.byref(ffi_p),
            ctypes.byref(c_ca),
            ctypes.byref(c_temp),
            ctypes.c_double(dt_min),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_step_rk4 failed with error code {rc}")
        return (c_ca.value, c_temp.value)

    def step_detailed(
        self,
        params: NativeCstrParams,
        ca: float,
        temp_k: float,
        dt_min: float = 0.005,
        runaway_threshold_k: float = 395.0,
        dt: Optional[float] = None,
    ) -> NativeCstrState:
        """
        Advances state by dt_min and populates conversion, reaction rate, and runaway status.
        """
        if dt is not None:
            dt_min = dt
        if not self._lib:
            new_ca, new_temp = self._py_step_rk4(params, ca, temp_k, dt_min)
            conv = (1.0 - (new_ca / params.ca_feed_mol_per_l)) * 100.0
            rate = self._py_reaction_rate(params, new_ca, new_temp)
            runaway = new_temp >= runaway_threshold_k
            return NativeCstrState(
                ca_mol_per_l=new_ca,
                temp_kelvin=new_temp,
                conversion_pct=conv,
                reaction_rate_mol_per_l_min=rate,
                is_runaway=runaway,
            )

        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_temp = ctypes.c_double(temp_k)
        c_state = CstrStateFFI()

        rc = self._lib.reactor_cstr_step_detailed(
            ctypes.byref(ffi_p),
            ctypes.byref(c_ca),
            ctypes.byref(c_temp),
            ctypes.c_double(dt_min),
            ctypes.c_double(runaway_threshold_k),
            ctypes.byref(c_state),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_step_detailed failed with error code {rc}")

        return NativeCstrState(
            ca_mol_per_l=c_state.ca_mol_per_l,
            temp_kelvin=c_state.temp_kelvin,
            conversion_pct=c_state.conversion_pct,
            reaction_rate_mol_per_l_min=c_state.reaction_rate_mol_per_l_min,
            is_runaway=c_state.is_runaway,
        )

    def simulate_trajectory(
        self,
        params: NativeCstrParams,
        ca0: float,
        temp0: float,
        dt_min: float,
        steps: int,
    ) -> TrajectoryResult:
        """
        Simulates an entire dynamic trajectory directly in native C with zero Python loop overhead.
        """
        if not self._lib:
            return self._py_simulate_trajectory(params, ca0, temp0, dt_min, steps)

        ffi_p = params.to_ffi()
        total_points = steps + 1

        array_type = ctypes.c_double * total_points
        out_time = array_type()
        out_ca = array_type()
        out_temp = array_type()

        rc = self._lib.reactor_cstr_simulate_trajectory(
            ctypes.byref(ffi_p),
            ctypes.c_double(ca0),
            ctypes.c_double(temp0),
            ctypes.c_double(dt_min),
            ctypes.c_size_t(steps),
            out_time,
            out_ca,
            out_temp,
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_simulate_trajectory failed with code {rc}")

        return TrajectoryResult(
            times_min=list(out_time),
            ca_mol_per_l=list(out_ca),
            temp_kelvin=list(out_temp),
        )

    # -------------------------------------------------------------------------
    # Pure Python Reference Implementation (for numerical parity validation)
    # -------------------------------------------------------------------------

    @staticmethod
    def _py_reaction_rate(p: NativeCstrParams, ca: float, temp_k: float) -> float:
        if ca <= 0.0 or temp_k <= 0.0:
            return 0.0
        k = p.k0_per_min * math.exp(-p.ea_over_r_kelvin / temp_k)
        return k * ca

    @classmethod
    def _py_derivatives(
        cls, p: NativeCstrParams, ca: float, temp_k: float
    ) -> Tuple[float, float]:
        eval_ca = max(0.0, ca)
        eval_t = max(50.0, temp_k)
        r = cls._py_reaction_rate(p, eval_ca, eval_t)

        space_time_inv = p.volumetric_flow_l_per_min / p.volume_liters
        heat_gen_coeff = p.delta_h_j_per_mol / p.rho_cp_j_per_l_k
        heat_rem_coeff = p.ua_j_per_min_k / (p.volume_liters * p.rho_cp_j_per_l_k)

        dca_dt = space_time_inv * (p.ca_feed_mol_per_l - eval_ca) - r
        dt_dt = (
            space_time_inv * (p.t_feed_kelvin - eval_t)
            + heat_gen_coeff * r
            - heat_rem_coeff * (eval_t - p.t_coolant_kelvin)
        )
        return (dca_dt, dt_dt)

    @classmethod
    def _py_step_rk4(
        cls, p: NativeCstrParams, ca: float, temp_k: float, dt: float
    ) -> Tuple[float, float]:
        half_dt = 0.5 * dt

        # Stage 1
        k1_ca, k1_t = cls._py_derivatives(p, ca, temp_k)

        # Stage 2
        y2_ca = ca + half_dt * k1_ca
        y2_t = temp_k + half_dt * k1_t
        k2_ca, k2_t = cls._py_derivatives(p, y2_ca, y2_t)

        # Stage 3
        y3_ca = ca + half_dt * k2_ca
        y3_t = temp_k + half_dt * k2_t
        k3_ca, k3_t = cls._py_derivatives(p, y3_ca, y3_t)

        # Stage 4
        y4_ca = ca + dt * k3_ca
        y4_t = temp_k + dt * k3_t
        k4_ca, k4_t = cls._py_derivatives(p, y4_ca, y4_t)

        # Assembly
        dt_six = dt / 6.0
        new_ca = ca + dt_six * (k1_ca + 2.0 * k2_ca + 2.0 * k3_ca + k4_ca)
        new_temp = temp_k + dt_six * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
        return (new_ca, new_temp)

    @classmethod
    def _py_simulate_trajectory(
        cls, p: NativeCstrParams, ca0: float, temp0: float, dt: float, steps: int
    ) -> TrajectoryResult:
        times = [0.0]
        cas = [ca0]
        temps = [temp0]

        curr_t = 0.0
        curr_ca = ca0
        curr_temp = temp0

        for _ in range(steps):
            curr_ca, curr_temp = cls._py_step_rk4(p, curr_ca, curr_temp, dt)
            curr_t += dt
            times.append(curr_t)
            cas.append(curr_ca)
            temps.append(curr_temp)

        return TrajectoryResult(times_min=times, ca_mol_per_l=cas, temp_kelvin=temps)
