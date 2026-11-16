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
from typing import List, Optional, Tuple, Sequence, Union



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


@dataclass
class BatchTrajectoryResult:
    times_min: List[float]
    ca_matrix: List[List[float]]  # [step][reactor]
    temp_matrix: List[List[float]]  # [step][reactor]


class CstrSeriesParallelParamsFFI(ctypes.Structure):
    """
    C-ABI compatible layout matching CstrSeriesParallelParamsFFI in ReactorCore.C/include/reactor_ffi.h.
    Total size: 176 bytes (22 x 64-bit IEEE-754 doubles).
    """
    _fields_ = [
        ("volumetric_flow_l_per_min", ctypes.c_double),
        ("volume_liters", ctypes.c_double),
        ("rho_cp_j_per_l_k", ctypes.c_double),
        ("k0_1_per_min", ctypes.c_double),
        ("ea1_over_r_kelvin", ctypes.c_double),
        ("delta_h1_j_per_mol", ctypes.c_double),
        ("k0_2_per_min", ctypes.c_double),
        ("ea2_over_r_kelvin", ctypes.c_double),
        ("delta_h2_j_per_mol", ctypes.c_double),
        ("k0_3_per_min", ctypes.c_double),
        ("ea3_over_r_kelvin", ctypes.c_double),
        ("delta_h3_j_per_mol", ctypes.c_double),
        ("ca_feed_mol_per_l", ctypes.c_double),
        ("cb_feed_mol_per_l", ctypes.c_double),
        ("cc_feed_mol_per_l", ctypes.c_double),
        ("cd_feed_mol_per_l", ctypes.c_double),
        ("t_feed_kelvin", ctypes.c_double),
        ("jacket_volume_liters", ctypes.c_double),
        ("jacket_flow_l_per_min", ctypes.c_double),
        ("jacket_rho_cp_j_per_l_k", ctypes.c_double),
        ("jacket_t_in_kelvin", ctypes.c_double),
        ("ua_j_per_min_k", ctypes.c_double),
    ]


class CstrSeriesParallelStateFFI(ctypes.Structure):
    """
    C-ABI compatible layout matching CstrSeriesParallelStateFFI in ReactorCore.C/include/reactor_ffi.h.
    """
    _fields_ = [
        ("ca_mol_per_l", ctypes.c_double),
        ("cb_mol_per_l", ctypes.c_double),
        ("cc_mol_per_l", ctypes.c_double),
        ("cd_mol_per_l", ctypes.c_double),
        ("temp_kelvin", ctypes.c_double),
        ("jacket_temp_kelvin", ctypes.c_double),
        ("conversion_a_pct", ctypes.c_double),
        ("yield_b_pct", ctypes.c_double),
        ("selectivity_b_pct", ctypes.c_double),
        ("rate_r1_mol_per_l_min", ctypes.c_double),
        ("rate_r2_mol_per_l_min", ctypes.c_double),
        ("rate_r3_mol_per_l_min", ctypes.c_double),
        ("heat_generated_watts", ctypes.c_double),
        ("heat_removed_watts", ctypes.c_double),
        ("is_thermal_runaway", ctypes.c_bool),
    ]


@dataclass
class NativeCstrSeriesParallelParams:
    volumetric_flow_l_per_min: float = 100.0
    volume_liters: float = 100.0
    rho_cp_j_per_l_k: float = 4000.0

    k0_1_per_min: float = 7.08e10
    ea1_over_r_kelvin: float = 8330.1
    delta_h1_j_per_mol: float = 200000.0

    k0_2_per_min: float = 3.50e9
    ea2_over_r_kelvin: float = 7500.0
    delta_h2_j_per_mol: float = 120000.0

    k0_3_per_min: float = 1.20e9
    ea3_over_r_kelvin: float = 9000.0
    delta_h3_j_per_mol: float = 150000.0

    ca_feed_mol_per_l: float = 1.0
    cb_feed_mol_per_l: float = 0.0
    cc_feed_mol_per_l: float = 0.0
    cd_feed_mol_per_l: float = 0.0
    t_feed_kelvin: float = 350.0

    jacket_volume_liters: float = 25.0
    jacket_flow_l_per_min: float = 50.0
    jacket_rho_cp_j_per_l_k: float = 4184.0
    jacket_t_in_kelvin: float = 295.0
    ua_j_per_min_k: float = 40000.0

    def to_ffi(self) -> CstrSeriesParallelParamsFFI:
        return CstrSeriesParallelParamsFFI(
            volumetric_flow_l_per_min=self.volumetric_flow_l_per_min,
            volume_liters=self.volume_liters,
            rho_cp_j_per_l_k=self.rho_cp_j_per_l_k,
            k0_1_per_min=self.k0_1_per_min,
            ea1_over_r_kelvin=self.ea1_over_r_kelvin,
            delta_h1_j_per_mol=self.delta_h1_j_per_mol,
            k0_2_per_min=self.k0_2_per_min,
            ea2_over_r_kelvin=self.ea2_over_r_kelvin,
            delta_h2_j_per_mol=self.delta_h2_j_per_mol,
            k0_3_per_min=self.k0_3_per_min,
            ea3_over_r_kelvin=self.ea3_over_r_kelvin,
            delta_h3_j_per_mol=self.delta_h3_j_per_mol,
            ca_feed_mol_per_l=self.ca_feed_mol_per_l,
            cb_feed_mol_per_l=self.cb_feed_mol_per_l,
            cc_feed_mol_per_l=self.cc_feed_mol_per_l,
            cd_feed_mol_per_l=self.cd_feed_mol_per_l,
            t_feed_kelvin=self.t_feed_kelvin,
            jacket_volume_liters=self.jacket_volume_liters,
            jacket_flow_l_per_min=self.jacket_flow_l_per_min,
            jacket_rho_cp_j_per_l_k=self.jacket_rho_cp_j_per_l_k,
            jacket_t_in_kelvin=self.jacket_t_in_kelvin,
            ua_j_per_min_k=self.ua_j_per_min_k,
        )

    @classmethod
    def from_ffi(cls, ffi: CstrSeriesParallelParamsFFI) -> NativeCstrSeriesParallelParams:
        return cls(
            volumetric_flow_l_per_min=ffi.volumetric_flow_l_per_min,
            volume_liters=ffi.volume_liters,
            rho_cp_j_per_l_k=ffi.rho_cp_j_per_l_k,
            k0_1_per_min=ffi.k0_1_per_min,
            ea1_over_r_kelvin=ffi.ea1_over_r_kelvin,
            delta_h1_j_per_mol=ffi.delta_h1_j_per_mol,
            k0_2_per_min=ffi.k0_2_per_min,
            ea2_over_r_kelvin=ffi.ea2_over_r_kelvin,
            delta_h2_j_per_mol=ffi.delta_h2_j_per_mol,
            k0_3_per_min=ffi.k0_3_per_min,
            ea3_over_r_kelvin=ffi.ea3_over_r_kelvin,
            delta_h3_j_per_mol=ffi.delta_h3_j_per_mol,
            ca_feed_mol_per_l=ffi.ca_feed_mol_per_l,
            cb_feed_mol_per_l=ffi.cb_feed_mol_per_l,
            cc_feed_mol_per_l=ffi.cc_feed_mol_per_l,
            cd_feed_mol_per_l=ffi.cd_feed_mol_per_l,
            t_feed_kelvin=ffi.t_feed_kelvin,
            jacket_volume_liters=ffi.jacket_volume_liters,
            jacket_flow_l_per_min=ffi.jacket_flow_l_per_min,
            jacket_rho_cp_j_per_l_k=ffi.jacket_rho_cp_j_per_l_k,
            jacket_t_in_kelvin=ffi.jacket_t_in_kelvin,
            ua_j_per_min_k=ffi.ua_j_per_min_k,
        )


@dataclass
class NativeCstrSeriesParallelState:
    ca_mol_per_l: float
    cb_mol_per_l: float
    cc_mol_per_l: float
    cd_mol_per_l: float
    temp_kelvin: float
    jacket_temp_kelvin: float
    conversion_a_pct: float
    yield_b_pct: float
    selectivity_b_pct: float
    rate_r1_mol_per_l_min: float
    rate_r2_mol_per_l_min: float
    rate_r3_mol_per_l_min: float
    heat_generated_watts: float
    heat_removed_watts: float
    is_thermal_runaway: bool


@dataclass
class SeriesParallelTrajectoryResult:
    times_min: List[float]
    ca_mol_per_l: List[float]
    cb_mol_per_l: List[float]
    cc_mol_per_l: List[float]
    cd_mol_per_l: List[float]
    temp_kelvin: List[float]
    jacket_temp_kelvin: List[float]


@dataclass
class SeriesParallelBatchTrajectoryResult:
    times_min: List[float]
    ca_matrix: List[List[float]]
    cb_matrix: List[List[float]]
    cc_matrix: List[List[float]]
    cd_matrix: List[List[float]]
    temp_matrix: List[List[float]]
    jacket_temp_matrix: List[List[float]]


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

            # Bind: int reactor_core_has_simd(void)
            if hasattr(lib, "reactor_core_has_simd"):
                lib.reactor_core_has_simd.argtypes = []
                lib.reactor_core_has_simd.restype = ctypes.c_int

            # Bind: int reactor_cstr_batch_step_rk4(...)
            if hasattr(lib, "reactor_cstr_batch_step_rk4"):
                lib.reactor_cstr_batch_step_rk4.argtypes = [
                    ctypes.POINTER(CstrParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                ]
                lib.reactor_cstr_batch_step_rk4.restype = ctypes.c_int

            # Bind: int reactor_cstr_batch_step_detailed(...)
            if hasattr(lib, "reactor_cstr_batch_step_detailed"):
                lib.reactor_cstr_batch_step_detailed.argtypes = [
                    ctypes.POINTER(CstrParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(CstrStateFFI),
                ]
                lib.reactor_cstr_batch_step_detailed.restype = ctypes.c_int

            # Bind: int reactor_cstr_batch_simulate_trajectory(...)
            if hasattr(lib, "reactor_cstr_batch_simulate_trajectory"):
                lib.reactor_cstr_batch_simulate_trajectory.argtypes = [
                    ctypes.POINTER(CstrParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                ]
                lib.reactor_cstr_batch_simulate_trajectory.restype = ctypes.c_int

            # Bind: int reactor_core_simd_level(void)
            if hasattr(lib, "reactor_core_simd_level"):
                lib.reactor_core_simd_level.argtypes = []
                lib.reactor_core_simd_level.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_default_params(CstrSeriesParallelParamsFFI *out_params)
            if hasattr(lib, "reactor_cstr_series_parallel_default_params"):
                lib.reactor_cstr_series_parallel_default_params.argtypes = [ctypes.POINTER(CstrSeriesParallelParamsFFI)]
                lib.reactor_cstr_series_parallel_default_params.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_rates(...)
            if hasattr(lib, "reactor_cstr_series_parallel_rates"):
                lib.reactor_cstr_series_parallel_rates.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                ]
                lib.reactor_cstr_series_parallel_rates.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_step_rk4(...)
            if hasattr(lib, "reactor_cstr_series_parallel_step_rk4"):
                lib.reactor_cstr_series_parallel_step_rk4.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                ]
                lib.reactor_cstr_series_parallel_step_rk4.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_step_detailed(...)
            if hasattr(lib, "reactor_cstr_series_parallel_step_detailed"):
                lib.reactor_cstr_series_parallel_step_detailed.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(CstrSeriesParallelStateFFI),
                ]
                lib.reactor_cstr_series_parallel_step_detailed.restype = ctypes.c_int

            if hasattr(lib, "reactor_cstr_step_ros3"):
                lib.reactor_cstr_step_ros3.argtypes = [
                    ctypes.POINTER(CstrParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                ]
                lib.reactor_cstr_step_ros3.restype = ctypes.c_int

            if hasattr(lib, "reactor_cstr_series_parallel_step_ros3"):
                lib.reactor_cstr_series_parallel_step_ros3.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                ]
                lib.reactor_cstr_series_parallel_step_ros3.restype = ctypes.c_int

            if hasattr(lib, "reactor_cstr_series_parallel_step_ros3_detailed"):
                lib.reactor_cstr_series_parallel_step_ros3_detailed.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(CstrSeriesParallelStateFFI),
                ]
                lib.reactor_cstr_series_parallel_step_ros3_detailed.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_batch_step_rk4(...)
            if hasattr(lib, "reactor_cstr_series_parallel_batch_step_rk4"):
                lib.reactor_cstr_series_parallel_batch_step_rk4.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                ]
                lib.reactor_cstr_series_parallel_batch_step_rk4.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_batch_step_detailed(...)
            if hasattr(lib, "reactor_cstr_series_parallel_batch_step_detailed"):
                lib.reactor_cstr_series_parallel_batch_step_detailed.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(CstrSeriesParallelStateFFI),
                ]
                lib.reactor_cstr_series_parallel_batch_step_detailed.restype = ctypes.c_int

            # Bind: int reactor_cstr_series_parallel_batch_simulate_trajectory(...)
            if hasattr(lib, "reactor_cstr_series_parallel_batch_simulate_trajectory"):
                lib.reactor_cstr_series_parallel_batch_simulate_trajectory.argtypes = [
                    ctypes.POINTER(CstrSeriesParallelParamsFFI),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                ]
                lib.reactor_cstr_series_parallel_batch_simulate_trajectory.restype = ctypes.c_int

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

    def has_simd(self) -> bool:
        """Returns True if the underlying C library was compiled with AVX2/FMA support."""
        if not self._lib or not hasattr(self._lib, "reactor_core_has_simd"):
            return False
        return self._lib.reactor_core_has_simd() == 1

    def step_batch_rk4(
        self,
        params: Union[NativeCstrParams, Sequence[NativeCstrParams]],
        ca_list: List[float],
        temp_k_list: List[float],
        dt_min: float = 0.005,
    ) -> Tuple[List[float], List[float]]:
        """
        Advances an ensemble of N CSTR reactors simultaneously using SIMD AVX2/FMA vectorization.
        Returns: (new_ca_list, new_temp_k_list).
        """
        n = len(ca_list)
        if n == 0 or len(temp_k_list) != n:
            raise ValueError("Concentration and temperature lists must have equal non-zero length.")

        if not self._lib or not hasattr(self._lib, "reactor_cstr_batch_step_rk4"):
            return self._py_step_batch_rk4(params, ca_list, temp_k_list, dt_min)

        p_obj = params[0] if isinstance(params, (list, tuple)) else params
        ffi_p = p_obj.to_ffi()
        array_type = ctypes.c_double * n
        c_ca = array_type(*ca_list)
        c_temp = array_type(*temp_k_list)

        rc = self._lib.reactor_cstr_batch_step_rk4(
            ctypes.byref(ffi_p),
            c_ca,
            c_temp,
            ctypes.c_size_t(n),
            ctypes.c_double(dt_min),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_batch_step_rk4 failed with error code {rc}")

        return (list(c_ca), list(c_temp))

    def step_batch_detailed(
        self,
        params: Union[NativeCstrParams, Sequence[NativeCstrParams]],
        ca_list: List[float],
        temp_k_list: List[float],
        dt_min: float = 0.005,
        runaway_threshold_k: float = 400.0,
    ) -> List[NativeCstrState]:
        """
        Advances N reactors simultaneously and evaluates detailed conversion, reaction rate, and runaway status.
        """
        n = len(ca_list)
        if n == 0 or len(temp_k_list) != n:
            raise ValueError("Arrays must have matching non-zero dimensions.")

        if not self._lib or not hasattr(self._lib, "reactor_cstr_batch_step_detailed"):
            return self._py_step_batch_detailed(params, ca_list, temp_k_list, dt_min, runaway_threshold_k)

        p_obj = params[0] if isinstance(params, (list, tuple)) else params
        ffi_p = p_obj.to_ffi()
        double_array_type = ctypes.c_double * n
        c_ca = double_array_type(*ca_list)
        c_temp = double_array_type(*temp_k_list)

        state_array_type = CstrStateFFI * n
        c_states = state_array_type()

        rc = self._lib.reactor_cstr_batch_step_detailed(
            ctypes.byref(ffi_p),
            c_ca,
            c_temp,
            ctypes.c_size_t(n),
            ctypes.c_double(dt_min),
            ctypes.c_double(runaway_threshold_k),
            c_states,
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_batch_step_detailed failed with error code {rc}")

        results: List[NativeCstrState] = []
        for i in range(n):
            s = c_states[i]
            results.append(NativeCstrState(
                ca_mol_per_l=s.ca_mol_per_l,
                temp_kelvin=s.temp_kelvin,
                conversion_pct=s.conversion_pct,
                reaction_rate_mol_per_l_min=s.reaction_rate_mol_per_l_min,
                is_runaway=s.is_runaway,
            ))
        return results

    def simulate_batch_trajectory(
        self,
        params: Union[NativeCstrParams, Sequence[NativeCstrParams]],
        ca0_list: List[float],
        temp0_list: List[float],
        dt_min: float,
        steps: int,
    ) -> BatchTrajectoryResult:
        """
        Simulates dynamic trajectories for N reactors in parallel entirely in native C (AVX2/FMA).
        """
        n = len(ca0_list)
        if n == 0 or len(temp0_list) != n:
            raise ValueError("Initial state arrays must have equal length.")

        if not self._lib or not hasattr(self._lib, "reactor_cstr_batch_simulate_trajectory"):
            return self._py_simulate_batch_trajectory(params, ca0_list, temp0_list, dt_min, steps)

        p_obj = params[0] if isinstance(params, (list, tuple)) else params
        ffi_p = p_obj.to_ffi()
        total_steps = steps + 1

        in_array_type = ctypes.c_double * n
        c_ca0 = in_array_type(*ca0_list)
        c_temp0 = in_array_type(*temp0_list)

        time_array_type = ctypes.c_double * total_steps
        out_time = time_array_type()

        traj_array_type = ctypes.c_double * (total_steps * n)
        out_ca = traj_array_type()
        out_temp = traj_array_type()

        rc = self._lib.reactor_cstr_batch_simulate_trajectory(
            ctypes.byref(ffi_p),
            c_ca0,
            c_temp0,
            ctypes.c_size_t(n),
            ctypes.c_double(dt_min),
            ctypes.c_size_t(steps),
            out_time,
            out_ca,
            out_temp,
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_batch_simulate_trajectory failed with code {rc}")

        times = list(out_time)
        ca_matrix: List[List[float]] = []
        temp_matrix: List[List[float]] = []

        for s in range(total_steps):
            offset = s * n
            ca_matrix.append(list(out_ca[offset:offset + n]))
            temp_matrix.append(list(out_temp[offset:offset + n]))

        return BatchTrajectoryResult(
            times_min=times,
            ca_matrix=ca_matrix,
            temp_matrix=temp_matrix,
        )

    def get_simd_level(self) -> int:
        """Returns the compiled SIMD instruction width: 512 for AVX-512, 256 for AVX2, 0 for scalar."""
        if not self._lib or not hasattr(self._lib, "reactor_core_simd_level"):
            return 0
        return int(self._lib.reactor_core_simd_level())

    def get_series_parallel_default_params(self) -> NativeCstrSeriesParallelParams:
        """Retrieves default parameters for series-parallel kinetics and dynamic jacket."""
        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_default_params"):
            return NativeCstrSeriesParallelParams()
        ffi_p = CstrSeriesParallelParamsFFI()
        rc = self._lib.reactor_cstr_series_parallel_default_params(ctypes.byref(ffi_p))
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_default_params failed with code {rc}")
        return NativeCstrSeriesParallelParams.from_ffi(ffi_p)

    def series_parallel_rates(
        self, params: NativeCstrSeriesParallelParams, ca: float, cb: float, temp_k: float
    ) -> Tuple[float, float, float]:
        """Calculates instantaneous reaction rates (r1, r2, r3) for series-parallel kinetics."""
        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_rates"):
            return self._py_series_parallel_rates(params, ca, cb, temp_k)
        ffi_p = params.to_ffi()
        c_r1 = ctypes.c_double()
        c_r2 = ctypes.c_double()
        c_r3 = ctypes.c_double()
        rc = self._lib.reactor_cstr_series_parallel_rates(
            ctypes.byref(ffi_p),
            ctypes.c_double(ca),
            ctypes.c_double(cb),
            ctypes.c_double(temp_k),
            ctypes.byref(c_r1),
            ctypes.byref(c_r2),
            ctypes.byref(c_r3),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_rates failed with code {rc}")
        return (c_r1.value, c_r2.value, c_r3.value)

    def step_series_parallel_rk4(
        self,
        params: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp_k: float,
        temp_j_k: float,
        dt_min: float = 0.005,
        dt: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float, float]:
        """Advances 6-variable state [Ca, Cb, Cc, Cd, T, Tj] by dt_min using RK4."""
        if dt is not None:
            dt_min = dt
        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_step_rk4"):
            return self._py_series_parallel_step_rk4(params, ca, cb, cc, cd, temp_k, temp_j_k, dt_min)
        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_cb = ctypes.c_double(cb)
        c_cc = ctypes.c_double(cc)
        c_cd = ctypes.c_double(cd)
        c_temp = ctypes.c_double(temp_k)
        c_temp_j = ctypes.c_double(temp_j_k)
        rc = self._lib.reactor_cstr_series_parallel_step_rk4(
            ctypes.byref(ffi_p),
            ctypes.byref(c_ca),
            ctypes.byref(c_cb),
            ctypes.byref(c_cc),
            ctypes.byref(c_cd),
            ctypes.byref(c_temp),
            ctypes.byref(c_temp_j),
            ctypes.c_double(dt_min),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_step_rk4 failed with code {rc}")
        return (c_ca.value, c_cb.value, c_cc.value, c_cd.value, c_temp.value, c_temp_j.value)

    def step_series_parallel_detailed(
        self,
        params: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp_k: float,
        temp_j_k: float,
        dt_min: float = 0.005,
        runaway_threshold_k: float = 400.0,
        dt: Optional[float] = None,
    ) -> NativeCstrSeriesParallelState:
        """Advances 6-variable state and calculates conversion, yield, selectivity, heat rates and runaway."""
        if dt is not None:
            dt_min = dt
        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_step_detailed"):
            return self._py_series_parallel_step_detailed(
                params, ca, cb, cc, cd, temp_k, temp_j_k, dt_min, runaway_threshold_k
            )
        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_cb = ctypes.c_double(cb)
        c_cc = ctypes.c_double(cc)
        c_cd = ctypes.c_double(cd)
        c_temp = ctypes.c_double(temp_k)
        c_temp_j = ctypes.c_double(temp_j_k)
        c_state = CstrSeriesParallelStateFFI()
        rc = self._lib.reactor_cstr_series_parallel_step_detailed(
            ctypes.byref(ffi_p),
            ctypes.byref(c_ca),
            ctypes.byref(c_cb),
            ctypes.byref(c_cc),
            ctypes.byref(c_cd),
            ctypes.byref(c_temp),
            ctypes.byref(c_temp_j),
            ctypes.c_double(dt_min),
            ctypes.c_double(runaway_threshold_k),
            ctypes.byref(c_state),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_step_detailed failed with code {rc}")
        return NativeCstrSeriesParallelState(
            ca_mol_per_l=c_state.ca_mol_per_l,
            cb_mol_per_l=c_state.cb_mol_per_l,
            cc_mol_per_l=c_state.cc_mol_per_l,
            cd_mol_per_l=c_state.cd_mol_per_l,
            temp_kelvin=c_state.temp_kelvin,
            jacket_temp_kelvin=c_state.jacket_temp_kelvin,
            conversion_a_pct=c_state.conversion_a_pct,
            yield_b_pct=c_state.yield_b_pct,
            selectivity_b_pct=c_state.selectivity_b_pct,
            rate_r1_mol_per_l_min=c_state.rate_r1_mol_per_l_min,
            rate_r2_mol_per_l_min=c_state.rate_r2_mol_per_l_min,
            rate_r3_mol_per_l_min=c_state.rate_r3_mol_per_l_min,
            heat_generated_watts=c_state.heat_generated_watts,
            heat_removed_watts=c_state.heat_removed_watts,
            is_thermal_runaway=c_state.is_thermal_runaway,
        )

    def step_stiff_ros3(
        self,
        params: NativeCstrParams,
        ca: float,
        temp_k: float,
        dt_min: float = 0.01,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Advances 2-variable single-reaction CSTR using L-stable 3rd-order Rosenbrock (ROS3L)."""
        if dt is not None:
            dt_min = dt
        if not self._lib or not hasattr(self._lib, "reactor_cstr_step_ros3"):
            return self._py_step_ros3(params, ca, temp_k, dt_min)
        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_temp = ctypes.c_double(temp_k)
        rc = self._lib.reactor_cstr_step_ros3(
            ctypes.byref(ffi_p), ctypes.byref(c_ca), ctypes.byref(c_temp), ctypes.c_double(dt_min)
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_step_ros3 failed with code {rc}")
        return (c_ca.value, c_temp.value)

    def step_series_parallel_ros3(
        self,
        params: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp_k: float,
        temp_j_k: float,
        dt_min: float = 0.01,
        dt: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float, float]:
        """Advances 6-variable series-parallel CSTR using single fixed L-stable ROS3L step (exact 6x6 Jacobian)."""
        if dt is not None:
            dt_min = dt
        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_step_ros3"):
            return self._py_series_parallel_step_ros3(params, ca, cb, cc, cd, temp_k, temp_j_k, dt_min)
        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_cb = ctypes.c_double(cb)
        c_cc = ctypes.c_double(cc)
        c_cd = ctypes.c_double(cd)
        c_temp = ctypes.c_double(temp_k)
        c_temp_j = ctypes.c_double(temp_j_k)
        rc = self._lib.reactor_cstr_series_parallel_step_ros3(
            ctypes.byref(ffi_p),
            ctypes.byref(c_ca),
            ctypes.byref(c_cb),
            ctypes.byref(c_cc),
            ctypes.byref(c_cd),
            ctypes.byref(c_temp),
            ctypes.byref(c_temp_j),
            ctypes.c_double(dt_min),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_step_ros3 failed with code {rc}")
        return (c_ca.value, c_cb.value, c_cc.value, c_cd.value, c_temp.value, c_temp_j.value)

    def step_series_parallel_ros3_detailed(
        self,
        params: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp_k: float,
        temp_j_k: float,
        dt_min: float = 0.01,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        runaway_threshold_k: float = 400.0,
        dt: Optional[float] = None,
    ) -> NativeCstrSeriesParallelState:
        """Advances 6-variable series-parallel CSTR via adaptive ROS3L(2) stiff integration and returns detailed metrics."""
        if dt is not None:
            dt_min = dt
        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_step_ros3_detailed"):
            return self._py_series_parallel_step_ros3_detailed(
                params, ca, cb, cc, cd, temp_k, temp_j_k, dt_min, rtol, atol, runaway_threshold_k
            )
        ffi_p = params.to_ffi()
        c_ca = ctypes.c_double(ca)
        c_cb = ctypes.c_double(cb)
        c_cc = ctypes.c_double(cc)
        c_cd = ctypes.c_double(cd)
        c_temp = ctypes.c_double(temp_k)
        c_temp_j = ctypes.c_double(temp_j_k)
        c_state = CstrSeriesParallelStateFFI()
        rc = self._lib.reactor_cstr_series_parallel_step_ros3_detailed(
            ctypes.byref(ffi_p),
            ctypes.byref(c_ca),
            ctypes.byref(c_cb),
            ctypes.byref(c_cc),
            ctypes.byref(c_cd),
            ctypes.byref(c_temp),
            ctypes.byref(c_temp_j),
            ctypes.c_double(dt_min),
            ctypes.c_double(rtol),
            ctypes.c_double(atol),
            ctypes.c_double(runaway_threshold_k),
            ctypes.byref(c_state),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_step_ros3_detailed failed with code {rc}")
        return NativeCstrSeriesParallelState(
            ca_mol_per_l=c_state.ca_mol_per_l,
            cb_mol_per_l=c_state.cb_mol_per_l,
            cc_mol_per_l=c_state.cc_mol_per_l,
            cd_mol_per_l=c_state.cd_mol_per_l,
            temp_kelvin=c_state.temp_kelvin,
            jacket_temp_kelvin=c_state.jacket_temp_kelvin,
            conversion_a_pct=c_state.conversion_a_pct,
            yield_b_pct=c_state.yield_b_pct,
            selectivity_b_pct=c_state.selectivity_b_pct,
            rate_r1_mol_per_l_min=c_state.rate_r1_mol_per_l_min,
            rate_r2_mol_per_l_min=c_state.rate_r2_mol_per_l_min,
            rate_r3_mol_per_l_min=c_state.rate_r3_mol_per_l_min,
            heat_generated_watts=c_state.heat_generated_watts,
            heat_removed_watts=c_state.heat_removed_watts,
            is_thermal_runaway=c_state.is_thermal_runaway,
        )

    def step_series_parallel_batch_rk4(
        self,
        params: Union[NativeCstrSeriesParallelParams, Sequence[NativeCstrSeriesParallelParams]],
        ca_list: List[float],
        cb_list: List[float],
        cc_list: List[float],
        cd_list: List[float],
        temp_list: List[float],
        temp_j_list: List[float],
        dt_min: float = 0.005,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        """Advances N reactors simultaneously with SIMD AVX-512/AVX2 vectorization."""
        n = len(ca_list)
        if not (len(cb_list) == n == len(cc_list) == len(cd_list) == len(temp_list) == len(temp_j_list) and n > 0):
            raise ValueError("All state lists must have equal non-zero length.")

        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_batch_step_rk4"):
            return self._py_series_parallel_step_batch_rk4(
                params, ca_list, cb_list, cc_list, cd_list, temp_list, temp_j_list, dt_min
            )

        p_obj = params[0] if isinstance(params, (list, tuple)) else params
        ffi_p = p_obj.to_ffi()
        arr_t = ctypes.c_double * n
        c_ca = arr_t(*ca_list)
        c_cb = arr_t(*cb_list)
        c_cc = arr_t(*cc_list)
        c_cd = arr_t(*cd_list)
        c_temp = arr_t(*temp_list)
        c_temp_j = arr_t(*temp_j_list)

        rc = self._lib.reactor_cstr_series_parallel_batch_step_rk4(
            ctypes.byref(ffi_p),
            c_ca, c_cb, c_cc, c_cd, c_temp, c_temp_j,
            ctypes.c_size_t(n),
            ctypes.c_double(dt_min),
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_batch_step_rk4 failed with code {rc}")

        return (list(c_ca), list(c_cb), list(c_cc), list(c_cd), list(c_temp), list(c_temp_j))

    def step_series_parallel_batch_detailed(
        self,
        params: Union[NativeCstrSeriesParallelParams, Sequence[NativeCstrSeriesParallelParams]],
        ca_list: List[float],
        cb_list: List[float],
        cc_list: List[float],
        cd_list: List[float],
        temp_list: List[float],
        temp_j_list: List[float],
        dt_min: float = 0.005,
        runaway_threshold_k: float = 400.0,
    ) -> List[NativeCstrSeriesParallelState]:
        """Advances N reactors and evaluates detailed state, yield, selectivity, and thermal metrics."""
        n = len(ca_list)
        if not (len(cb_list) == n == len(cc_list) == len(cd_list) == len(temp_list) == len(temp_j_list) and n > 0):
            raise ValueError("All state lists must have equal non-zero length.")

        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_batch_step_detailed"):
            return self._py_series_parallel_step_batch_detailed(
                params, ca_list, cb_list, cc_list, cd_list, temp_list, temp_j_list, dt_min, runaway_threshold_k
            )

        p_obj = params[0] if isinstance(params, (list, tuple)) else params
        ffi_p = p_obj.to_ffi()
        arr_t = ctypes.c_double * n
        c_ca = arr_t(*ca_list)
        c_cb = arr_t(*cb_list)
        c_cc = arr_t(*cc_list)
        c_cd = arr_t(*cd_list)
        c_temp = arr_t(*temp_list)
        c_temp_j = arr_t(*temp_j_list)

        state_arr_t = CstrSeriesParallelStateFFI * n
        c_states = state_arr_t()

        rc = self._lib.reactor_cstr_series_parallel_batch_step_detailed(
            ctypes.byref(ffi_p),
            c_ca, c_cb, c_cc, c_cd, c_temp, c_temp_j,
            ctypes.c_size_t(n),
            ctypes.c_double(dt_min),
            ctypes.c_double(runaway_threshold_k),
            c_states,
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_batch_step_detailed failed with code {rc}")

        results: List[NativeCstrSeriesParallelState] = []
        for i in range(n):
            s = c_states[i]
            results.append(NativeCstrSeriesParallelState(
                ca_mol_per_l=s.ca_mol_per_l,
                cb_mol_per_l=s.cb_mol_per_l,
                cc_mol_per_l=s.cc_mol_per_l,
                cd_mol_per_l=s.cd_mol_per_l,
                temp_kelvin=s.temp_kelvin,
                jacket_temp_kelvin=s.jacket_temp_kelvin,
                conversion_a_pct=s.conversion_a_pct,
                yield_b_pct=s.yield_b_pct,
                selectivity_b_pct=s.selectivity_b_pct,
                rate_r1_mol_per_l_min=s.rate_r1_mol_per_l_min,
                rate_r2_mol_per_l_min=s.rate_r2_mol_per_l_min,
                rate_r3_mol_per_l_min=s.rate_r3_mol_per_l_min,
                heat_generated_watts=s.heat_generated_watts,
                heat_removed_watts=s.heat_removed_watts,
                is_thermal_runaway=s.is_thermal_runaway,
            ))
        return results

    def simulate_series_parallel_batch_trajectory(
        self,
        params: Union[NativeCstrSeriesParallelParams, Sequence[NativeCstrSeriesParallelParams]],
        ca0_list: List[float],
        cb0_list: List[float],
        cc0_list: List[float],
        cd0_list: List[float],
        temp0_list: List[float],
        temp_j0_list: List[float],
        dt_min: float,
        steps: int,
    ) -> SeriesParallelBatchTrajectoryResult:
        """Simulates parallel trajectories for N series-parallel reactors entirely in C."""
        n = len(ca0_list)
        if not (len(cb0_list) == n == len(cc0_list) == len(cd0_list) == len(temp0_list) == len(temp_j0_list) and n > 0):
            raise ValueError("All initial state arrays must have equal length.")

        if not self._lib or not hasattr(self._lib, "reactor_cstr_series_parallel_batch_simulate_trajectory"):
            return self._py_series_parallel_simulate_batch_trajectory(
                params, ca0_list, cb0_list, cc0_list, cd0_list, temp0_list, temp_j0_list, dt_min, steps
            )

        p_obj = params[0] if isinstance(params, (list, tuple)) else params
        ffi_p = p_obj.to_ffi()
        total_steps = steps + 1

        in_t = ctypes.c_double * n
        c_ca0 = in_t(*ca0_list)
        c_cb0 = in_t(*cb0_list)
        c_cc0 = in_t(*cc0_list)
        c_cd0 = in_t(*cd0_list)
        c_temp0 = in_t(*temp0_list)
        c_temp_j0 = in_t(*temp_j0_list)

        time_t = ctypes.c_double * total_steps
        out_time = time_t()

        traj_t = ctypes.c_double * (total_steps * n)
        out_ca = traj_t()
        out_cb = traj_t()
        out_cc = traj_t()
        out_cd = traj_t()
        out_temp = traj_t()
        out_temp_j = traj_t()

        rc = self._lib.reactor_cstr_series_parallel_batch_simulate_trajectory(
            ctypes.byref(ffi_p),
            c_ca0, c_cb0, c_cc0, c_cd0, c_temp0, c_temp_j0,
            ctypes.c_size_t(n),
            ctypes.c_double(dt_min),
            ctypes.c_size_t(steps),
            out_time,
            out_ca, out_cb, out_cc, out_cd, out_temp, out_temp_j,
        )
        if rc != 0:
            raise RuntimeError(f"reactor_cstr_series_parallel_batch_simulate_trajectory failed with code {rc}")

        times = list(out_time)
        ca_m: List[List[float]] = []
        cb_m: List[List[float]] = []
        cc_m: List[List[float]] = []
        cd_m: List[List[float]] = []
        temp_m: List[List[float]] = []
        temp_j_m: List[List[float]] = []

        for s in range(total_steps):
            offset = s * n
            ca_m.append(list(out_ca[offset:offset + n]))
            cb_m.append(list(out_cb[offset:offset + n]))
            cc_m.append(list(out_cc[offset:offset + n]))
            cd_m.append(list(out_cd[offset:offset + n]))
            temp_m.append(list(out_temp[offset:offset + n]))
            temp_j_m.append(list(out_temp_j[offset:offset + n]))

        return SeriesParallelBatchTrajectoryResult(
            times_min=times,
            ca_matrix=ca_m,
            cb_matrix=cb_m,
            cc_matrix=cc_m,
            cd_matrix=cd_m,
            temp_matrix=temp_m,
            jacket_temp_matrix=temp_j_m,
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

    @classmethod
    def _py_step_batch_rk4(
        cls, p: Union[NativeCstrParams, Sequence[NativeCstrParams]], ca_list: List[float], temp_list: List[float], dt: float
    ) -> Tuple[List[float], List[float]]:
        new_ca: List[float] = []
        new_temp: List[float] = []
        is_seq = isinstance(p, (list, tuple))
        for i, (c, t) in enumerate(zip(ca_list, temp_list)):
            p_i = p[i] if is_seq else p
            c_next, t_next = cls._py_step_rk4(p_i, c, t, dt)
            new_ca.append(c_next)
            new_temp.append(t_next)
        return (new_ca, new_temp)

    @classmethod
    def _py_step_batch_detailed(
        cls, p: Union[NativeCstrParams, Sequence[NativeCstrParams]], ca_list: List[float], temp_list: List[float], dt: float, threshold_k: float
    ) -> List[NativeCstrState]:
        states: List[NativeCstrState] = []
        is_seq = isinstance(p, (list, tuple))
        for i, (c, t) in enumerate(zip(ca_list, temp_list)):
            p_i = p[i] if is_seq else p
            c_next, t_next = cls._py_step_rk4(p_i, c, t, dt)
            if c_next < 0.0:
                c_next = 0.0
            conv = max(0.0, min(100.0, (1.0 - (c_next / p_i.ca_feed_mol_per_l)) * 100.0)) if p_i.ca_feed_mol_per_l > 0.0 else 0.0
            rate = cls._py_reaction_rate(p_i, c_next, t_next)
            states.append(NativeCstrState(
                ca_mol_per_l=c_next,
                temp_kelvin=t_next,
                conversion_pct=conv,
                reaction_rate_mol_per_l_min=rate,
                is_runaway=t_next >= threshold_k,
            ))
        return states

    @classmethod
    def _py_simulate_batch_trajectory(
        cls, p: Union[NativeCstrParams, Sequence[NativeCstrParams]], ca0_list: List[float], temp0_list: List[float], dt: float, steps: int
    ) -> BatchTrajectoryResult:
        times = [0.0]
        ca_matrix = [[float(x) for x in ca0_list]]
        temp_matrix = [[float(x) for x in temp0_list]]

        curr_ca = list(ca0_list)
        curr_temp = list(temp0_list)
        curr_t = 0.0

        for _ in range(steps):
            curr_ca, curr_temp = cls._py_step_batch_rk4(p, curr_ca, curr_temp, dt)
            curr_t += dt
            times.append(curr_t)
            ca_matrix.append(list(curr_ca))
            temp_matrix.append(list(curr_temp))

        return BatchTrajectoryResult(times_min=times, ca_matrix=ca_matrix, temp_matrix=temp_matrix)

    @staticmethod
    def _py_series_parallel_rates(
        p: NativeCstrSeriesParallelParams, ca: float, cb: float, temp_k: float
    ) -> Tuple[float, float, float]:
        if temp_k <= 0.0:
            return (0.0, 0.0, 0.0)
        ca_eval = max(0.0, ca)
        cb_eval = max(0.0, cb)
        t_eval = max(50.0, temp_k)
        k1 = p.k0_1_per_min * math.exp(-p.ea1_over_r_kelvin / t_eval)
        k2 = p.k0_2_per_min * math.exp(-p.ea2_over_r_kelvin / t_eval)
        k3 = p.k0_3_per_min * math.exp(-p.ea3_over_r_kelvin / t_eval)
        return (k1 * ca_eval, k2 * cb_eval, k3 * ca_eval)

    @classmethod
    def _py_series_parallel_derivatives(
        cls,
        p: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp: float,
        temp_j: float,
    ) -> Tuple[float, float, float, float, float, float]:
        ca_eval = max(0.0, ca)
        cb_eval = max(0.0, cb)
        cc_eval = max(0.0, cc)
        cd_eval = max(0.0, cd)
        t_eval = max(50.0, temp)
        tj_eval = max(50.0, temp_j)

        r1, r2, r3 = cls._py_series_parallel_rates(p, ca_eval, cb_eval, t_eval)

        space_time_inv = p.volumetric_flow_l_per_min / p.volume_liters
        qj_over_vj = p.jacket_flow_l_per_min / p.jacket_volume_liters

        heat_gen = (
            p.delta_h1_j_per_mol * r1
            + p.delta_h2_j_per_mol * r2
            + p.delta_h3_j_per_mol * r3
        ) / p.rho_cp_j_per_l_k

        heat_rem_reactor = (
            p.ua_j_per_min_k / (p.volume_liters * p.rho_cp_j_per_l_k)
        ) * (t_eval - tj_eval)
        heat_rem_jacket = (
            p.ua_j_per_min_k / (p.jacket_volume_liters * p.jacket_rho_cp_j_per_l_k)
        ) * (t_eval - tj_eval)

        dca_dt = space_time_inv * (p.ca_feed_mol_per_l - ca_eval) - r1 - r3
        dcb_dt = space_time_inv * (p.cb_feed_mol_per_l - cb_eval) + r1 - r2
        dcc_dt = space_time_inv * (p.cc_feed_mol_per_l - cc_eval) + r2
        dcd_dt = space_time_inv * (p.cd_feed_mol_per_l - cd_eval) + r3
        dt_dt = space_time_inv * (p.t_feed_kelvin - t_eval) + heat_gen - heat_rem_reactor
        dtj_dt = qj_over_vj * (p.jacket_t_in_kelvin - tj_eval) + heat_rem_jacket

        return (dca_dt, dcb_dt, dcc_dt, dcd_dt, dt_dt, dtj_dt)

    @classmethod
    def _py_series_parallel_step_rk4(
        cls,
        p: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp: float,
        temp_j: float,
        dt: float,
    ) -> Tuple[float, float, float, float, float, float]:
        half_dt = 0.5 * dt
        y0 = (ca, cb, cc, cd, temp, temp_j)

        # Stage 1
        k1 = cls._py_series_parallel_derivatives(p, *y0)

        # Stage 2
        y2 = tuple(y + half_dt * k for y, k in zip(y0, k1))
        k2 = cls._py_series_parallel_derivatives(p, *y2)

        # Stage 3
        y3 = tuple(y + half_dt * k for y, k in zip(y0, k2))
        k3 = cls._py_series_parallel_derivatives(p, *y3)

        # Stage 4
        y4 = tuple(y + dt * k for y, k in zip(y0, k3))
        k4 = cls._py_series_parallel_derivatives(p, *y4)

        dt_six = dt / 6.0
        y_next = [
            y + dt_six * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
            for y, k1_i, k2_i, k3_i, k4_i in zip(y0, k1, k2, k3, k4)
        ]

        # Clamps
        y_next[0] = max(0.0, y_next[0])
        y_next[1] = max(0.0, y_next[1])
        y_next[2] = max(0.0, y_next[2])
        y_next[3] = max(0.0, y_next[3])
        y_next[4] = max(50.0, y_next[4])
        y_next[5] = max(50.0, y_next[5])

        return (y_next[0], y_next[1], y_next[2], y_next[3], y_next[4], y_next[5])

    @classmethod
    def _py_series_parallel_step_detailed(
        cls,
        p: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp: float,
        temp_j: float,
        dt: float,
        runaway_threshold_k: float,
    ) -> NativeCstrSeriesParallelState:
        n_ca, n_cb, n_cc, n_cd, n_temp, n_temp_j = cls._py_series_parallel_step_rk4(
            p, ca, cb, cc, cd, temp, temp_j, dt
        )
        r1, r2, r3 = cls._py_series_parallel_rates(p, n_ca, n_cb, n_temp)

        ca0 = p.ca_feed_mol_per_l if p.ca_feed_mol_per_l > 1e-12 else 1.0
        conv_a = ((p.ca_feed_mol_per_l - n_ca) / ca0) * 100.0
        yield_b = ((n_cb - p.cb_feed_mol_per_l) / ca0) * 100.0

        total_prod = n_cb + n_cc + n_cd
        select_b = (n_cb / total_prod * 100.0) if total_prod > 1e-12 else 100.0

        heat_gen_j_per_min = p.volume_liters * (
            p.delta_h1_j_per_mol * r1
            + p.delta_h2_j_per_mol * r2
            + p.delta_h3_j_per_mol * r3
        )
        heat_gen_w = heat_gen_j_per_min / 60.0

        heat_rem_j_per_min = p.ua_j_per_min_k * (n_temp - n_temp_j)
        heat_rem_w = heat_rem_j_per_min / 60.0

        runaway = n_temp >= runaway_threshold_k

        return NativeCstrSeriesParallelState(
            ca_mol_per_l=n_ca,
            cb_mol_per_l=n_cb,
            cc_mol_per_l=n_cc,
            cd_mol_per_l=n_cd,
            temp_kelvin=n_temp,
            jacket_temp_kelvin=n_temp_j,
            conversion_a_pct=conv_a,
            yield_b_pct=yield_b,
            selectivity_b_pct=select_b,
            rate_r1_mol_per_l_min=r1,
            rate_r2_mol_per_l_min=r2,
            rate_r3_mol_per_l_min=r3,
            heat_generated_watts=heat_gen_w,
            heat_removed_watts=heat_rem_w,
            is_thermal_runaway=runaway,
        )

    @classmethod
    def _py_series_parallel_step_batch_rk4(
        cls,
        p: Union[NativeCstrSeriesParallelParams, Sequence[NativeCstrSeriesParallelParams]],
        ca_list: List[float],
        cb_list: List[float],
        cc_list: List[float],
        cd_list: List[float],
        temp_list: List[float],
        temp_j_list: List[float],
        dt: float,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        out_ca: List[float] = []
        out_cb: List[float] = []
        out_cc: List[float] = []
        out_cd: List[float] = []
        out_temp: List[float] = []
        out_temp_j: List[float] = []

        is_seq = isinstance(p, (list, tuple))
        for i in range(len(ca_list)):
            p_i = p[i] if is_seq else p
            c_a, c_b, c_c, c_d, t, tj = cls._py_series_parallel_step_rk4(
                p_i, ca_list[i], cb_list[i], cc_list[i], cd_list[i], temp_list[i], temp_j_list[i], dt
            )
            out_ca.append(c_a)
            out_cb.append(c_b)
            out_cc.append(c_c)
            out_cd.append(c_d)
            out_temp.append(t)
            out_temp_j.append(tj)

        return (out_ca, out_cb, out_cc, out_cd, out_temp, out_temp_j)

    @classmethod
    def _py_series_parallel_step_batch_detailed(
        cls,
        p: Union[NativeCstrSeriesParallelParams, Sequence[NativeCstrSeriesParallelParams]],
        ca_list: List[float],
        cb_list: List[float],
        cc_list: List[float],
        cd_list: List[float],
        temp_list: List[float],
        temp_j_list: List[float],
        dt: float,
        threshold_k: float,
    ) -> List[NativeCstrSeriesParallelState]:
        states: List[NativeCstrSeriesParallelState] = []
        is_seq = isinstance(p, (list, tuple))
        for i in range(len(ca_list)):
            p_i = p[i] if is_seq else p
            s = cls._py_series_parallel_step_detailed(
                p_i, ca_list[i], cb_list[i], cc_list[i], cd_list[i], temp_list[i], temp_j_list[i], dt, threshold_k
            )
            states.append(s)
        return states

    @classmethod
    def _py_series_parallel_simulate_batch_trajectory(
        cls,
        p: Union[NativeCstrSeriesParallelParams, Sequence[NativeCstrSeriesParallelParams]],
        ca0_list: List[float],
        cb0_list: List[float],
        cc0_list: List[float],
        cd0_list: List[float],
        temp0_list: List[float],
        temp_j0_list: List[float],
        dt: float,
        steps: int,
    ) -> SeriesParallelBatchTrajectoryResult:
        times = [0.0]
        ca_m = [[float(x) for x in ca0_list]]
        cb_m = [[float(x) for x in cb0_list]]
        cc_m = [[float(x) for x in cc0_list]]
        cd_m = [[float(x) for x in cd0_list]]
        temp_m = [[float(x) for x in temp0_list]]
        temp_j_m = [[float(x) for x in temp_j0_list]]

        curr_ca = list(ca0_list)
        curr_cb = list(cb0_list)
        curr_cc = list(cc0_list)
        curr_cd = list(cd0_list)
        curr_temp = list(temp0_list)
        curr_temp_j = list(temp_j0_list)
        curr_t = 0.0

        for _ in range(steps):
            curr_ca, curr_cb, curr_cc, curr_cd, curr_temp, curr_temp_j = cls._py_series_parallel_step_batch_rk4(
                p, curr_ca, curr_cb, curr_cc, curr_cd, curr_temp, curr_temp_j, dt
            )
            curr_t += dt
            times.append(curr_t)
            ca_m.append(list(curr_ca))
            cb_m.append(list(curr_cb))
            cc_m.append(list(curr_cc))
            cd_m.append(list(curr_cd))
            temp_m.append(list(curr_temp))
            temp_j_m.append(list(curr_temp_j))

        return SeriesParallelBatchTrajectoryResult(
            times_min=times,
            ca_matrix=ca_m,
            cb_matrix=cb_m,
            cc_matrix=cc_m,
            cd_matrix=cd_m,
            temp_matrix=temp_m,
            jacket_temp_matrix=temp_j_m,
        )

    _ROS3_GAMMA = 0.435866521508459
    _ROS3_A21 = 2.2942803602790418
    _ROS3_C21 = -7.5580027318411718
    _ROS3_C31 = 12.2256646616911144
    _ROS3_C32 = 5.7401972086621127
    _ROS3_M1 = 0.7515765852110374
    _ROS3_M2 = -0.5743194747305118
    _ROS3_M3 = 0.3823800600465069
    _ROS3_E1 = -0.7681987171119269
    _ROS3_E2 = -0.2367386492284888
    _ROS3_E3 = 0.3823800600465069

    @classmethod
    def _py_step_ros3(
        cls, p: NativeCstrParams, ca: float, temp_k: float, dt: float
    ) -> Tuple[float, float]:
        if dt <= 0.0 or p.volume_liters <= 0.0:
            raise ValueError("Invalid dt or reactor volume for ROS3L step")
        ca_v = max(0.0, ca)
        t_v = max(50.0, min(2500.0, temp_k))

        s = p.volumetric_flow_l_per_min / p.volume_liters
        cg = p.delta_h_j_per_mol / p.rho_cp_j_per_l_k
        cr = p.ua_j_per_min_k / (p.volume_liters * p.rho_cp_j_per_l_k)
        k = p.k0_per_min * math.exp(-p.ea_over_r_kelvin / t_v)
        dr_dt = ca_v * k * p.ea_over_r_kelvin / (t_v * t_v)

        inv_gh = 1.0 / (cls._ROS3_GAMMA * dt)
        w00 = inv_gh - (-s - k)
        w01 = dr_dt
        w10 = -(cg * k)
        w11 = inv_gh - (-s - cr + cg * dr_dt)
        det = w00 * w11 - w01 * w10
        if abs(det) < 1e-15 or not math.isfinite(det):
            raise RuntimeError("Singular 2x2 Jacobian in _py_step_ros3")
        inv_det = 1.0 / det

        def rhs2(c: float, t: float) -> Tuple[float, float]:
            cv = max(0.0, c)
            tv = max(50.0, min(2500.0, t))
            r = p.k0_per_min * math.exp(-p.ea_over_r_kelvin / tv) * cv
            return (
                s * (p.ca_feed_mol_per_l - cv) - r,
                s * (p.t_feed_kelvin - tv) + cg * r - cr * (tv - p.t_coolant_kelvin),
            )

        f10, f11 = rhs2(ca_v, t_v)
        w1_0 = (w11 * f10 - w01 * f11) * inv_det
        w1_1 = (-w10 * f10 + w00 * f11) * inv_det

        f20, f21 = rhs2(ca_v + cls._ROS3_A21 * w1_0, t_v + cls._ROS3_A21 * w1_1)
        inv_dt = 1.0 / dt
        r20 = f20 + cls._ROS3_C21 * inv_dt * w1_0
        r21 = f21 + cls._ROS3_C21 * inv_dt * w1_1
        w2_0 = (w11 * r20 - w01 * r21) * inv_det
        w2_1 = (-w10 * r20 + w00 * r21) * inv_det

        r30 = f20 + (cls._ROS3_C31 * w1_0 + cls._ROS3_C32 * w2_0) * inv_dt
        r31 = f21 + (cls._ROS3_C31 * w1_1 + cls._ROS3_C32 * w2_1) * inv_det
        w3_0 = (w11 * r30 - w01 * r31) * inv_det
        w3_1 = (-w10 * r30 + w00 * r31) * inv_det

        ca_next = max(0.0, ca_v + cls._ROS3_M1 * w1_0 + cls._ROS3_M2 * w2_0 + cls._ROS3_M3 * w3_0)
        t_next = max(50.0, min(2500.0, t_v + cls._ROS3_M1 * w1_1 + cls._ROS3_M2 * w2_1 + cls._ROS3_M3 * w3_1))
        return (ca_next, t_next)

    @staticmethod
    def _py_solve_lu6(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
        m = [row[:] for row in a]
        rhs = b[:]
        for k in range(6):
            max_row = k
            max_val = abs(m[k][k])
            for r in range(k + 1, 6):
                val = abs(m[r][k])
                if val > max_val:
                    max_val = val
                    max_row = r
            if max_val < 1e-15 or not math.isfinite(max_val):
                return None
            if max_row != k:
                m[k], m[max_row] = m[max_row], m[k]
                rhs[k], rhs[max_row] = rhs[max_row], rhs[k]
            inv_piv = 1.0 / m[k][k]
            for r in range(k + 1, 6):
                factor = m[r][k] * inv_piv
                for c in range(k + 1, 6):
                    m[r][c] -= factor * m[k][c]
                rhs[r] -= factor * rhs[k]
        x = [0.0] * 6
        for r in range(5, -1, -1):
            s = rhs[r]
            for c in range(r + 1, 6):
                s -= m[r][c] * x[c]
            x[r] = s / m[r][r]
        return x

    @classmethod
    def _py_series_parallel_ros3_single(
        cls,
        p: NativeCstrSeriesParallelParams,
        x_in: List[float],
        dt: float,
    ) -> Optional[Tuple[List[float], List[float]]]:
        ca = max(0.0, x_in[0])
        cb = max(0.0, x_in[1])
        t = max(50.0, min(2500.0, x_in[4]))

        k1 = p.k0_1_per_min * math.exp(-p.ea1_over_r_kelvin / t)
        k2 = p.k0_2_per_min * math.exp(-p.ea2_over_r_kelvin / t)
        k3 = p.k0_3_per_min * math.exp(-p.ea3_over_r_kelvin / t)
        inv_t2 = 1.0 / (t * t)
        dr1 = ca * k1 * p.ea1_over_r_kelvin * inv_t2
        dr2 = cb * k2 * p.ea2_over_r_kelvin * inv_t2
        dr3 = ca * k3 * p.ea3_over_r_kelvin * inv_t2

        s = p.volumetric_flow_l_per_min / p.volume_liters
        sj = p.jacket_flow_l_per_min / p.jacket_volume_liters
        c1 = p.delta_h1_j_per_mol / p.rho_cp_j_per_l_k
        c2 = p.delta_h2_j_per_mol / p.rho_cp_j_per_l_k
        c3 = p.delta_h3_j_per_mol / p.rho_cp_j_per_l_k
        hr = p.ua_j_per_min_k / (p.volume_liters * p.rho_cp_j_per_l_k)
        hj = p.ua_j_per_min_k / (p.jacket_volume_liters * p.jacket_rho_cp_j_per_l_k)

        inv_gh = 1.0 / (cls._ROS3_GAMMA * dt)
        w = [[inv_gh if i == j else 0.0 for j in range(6)] for i in range(6)]
        w[0][0] -= -s - k1 - k3
        w[0][4] -= -(dr1 + dr3)
        w[1][0] -= k1
        w[1][1] -= -s - k2
        w[1][4] -= dr1 - dr2
        w[2][1] -= k2
        w[2][2] -= -s
        w[2][4] -= dr2
        w[3][0] -= k3
        w[3][3] -= -s
        w[3][4] -= dr3
        w[4][0] -= c1 * k1 + c3 * k3
        w[4][1] -= c2 * k2
        w[4][4] -= -s - hr + c1 * dr1 + c2 * dr2 + c3 * dr3
        w[4][5] -= hr
        w[5][4] -= hj
        w[5][5] -= -sj - hj

        f1 = list(cls._py_series_parallel_derivatives(p, *x_in))
        w1 = cls._py_solve_lu6(w, f1)
        if w1 is None:
            return None

        x2 = [x_in[i] + cls._ROS3_A21 * w1[i] for i in range(6)]
        f2 = list(cls._py_series_parallel_derivatives(p, *x2))
        inv_dt = 1.0 / dt
        r2 = [f2[i] + cls._ROS3_C21 * inv_dt * w1[i] for i in range(6)]
        w2 = cls._py_solve_lu6(w, r2)
        if w2 is None:
            return None

        r3 = [f2[i] + (cls._ROS3_C31 * w1[i] + cls._ROS3_C32 * w2[i]) * inv_dt for i in range(6)]
        w3 = cls._py_solve_lu6(w, r3)
        if w3 is None:
            return None

        x_out = [
            x_in[i] + cls._ROS3_M1 * w1[i] + cls._ROS3_M2 * w2[i] + cls._ROS3_M3 * w3[i]
            for i in range(6)
        ]
        err = [
            cls._ROS3_E1 * w1[i] + cls._ROS3_E2 * w2[i] + cls._ROS3_E3 * w3[i]
            for i in range(6)
        ]
        if any(not math.isfinite(v) for v in x_out):
            return None
        return (x_out, err)

    @classmethod
    def _py_series_parallel_step_ros3(
        cls,
        p: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp_k: float,
        temp_j_k: float,
        dt: float,
    ) -> Tuple[float, float, float, float, float, float]:
        res = cls._py_series_parallel_ros3_single(p, [ca, cb, cc, cd, temp_k, temp_j_k], dt)
        if res is None:
            raise RuntimeError("Series-parallel Python ROS3L step failed")
        x_next, _ = res
        return (
            max(0.0, x_next[0]),
            max(0.0, x_next[1]),
            max(0.0, x_next[2]),
            max(0.0, x_next[3]),
            max(50.0, min(2500.0, x_next[4])),
            max(50.0, min(2500.0, x_next[5])),
        )

    @classmethod
    def _py_series_parallel_step_ros3_detailed(
        cls,
        p: NativeCstrSeriesParallelParams,
        ca: float,
        cb: float,
        cc: float,
        cd: float,
        temp_k: float,
        temp_j_k: float,
        dt: float,
        rtol: float,
        atol: float,
        threshold_k: float,
    ) -> NativeCstrSeriesParallelState:
        rel = rtol if rtol > 0.0 else 1e-5
        abs_tol = atol if atol > 0.0 else 1e-7
        x = [ca, cb, cc, cd, temp_k, temp_j_k]
        t = 0.0
        h = dt
        h_min = 1e-8
        for _ in range(4096):
            if t >= dt - 1e-12:
                break
            if t + h > dt:
                h = dt - t
            res = cls._py_series_parallel_ros3_single(p, x, h)
            if res is None:
                h *= 0.25
                if h < h_min:
                    raise RuntimeError("Adaptive Python ROS3L step size underflowed")
                continue
            x_cand, err_vec = res
            err_norm = math.sqrt(
                sum((abs(err_vec[i]) / (abs_tol + rel * max(abs(x[i]), abs(x_cand[i])))) ** 2 for i in range(6)) / 6.0
            )
            if err_norm <= 1.0 or h <= h_min * 1.1:
                x = [
                    max(0.0, x_cand[0]),
                    max(0.0, x_cand[1]),
                    max(0.0, x_cand[2]),
                    max(0.0, x_cand[3]),
                    max(50.0, min(2500.0, x_cand[4])),
                    max(50.0, min(2500.0, x_cand[5])),
                ]
                t += h
                fac = 2.0 if err_norm < 1e-6 else min(2.0, max(0.5, 0.9 * (err_norm ** (-1.0 / 3.0))))
                h *= fac
            else:
                h *= max(0.2, min(0.7, 0.9 * (err_norm ** (-1.0 / 3.0))))

        n_ca, n_cb, n_cc, n_cd, n_temp, n_temp_j = x
        r1, r2, r3 = cls._py_series_parallel_rates(p, n_ca, n_cb, n_temp)
        ca0 = p.ca_feed_mol_per_l if p.ca_feed_mol_per_l > 1e-12 else 1.0
        conv_a = ((p.ca_feed_mol_per_l - n_ca) / ca0) * 100.0
        yield_b = ((n_cb - p.cb_feed_mol_per_l) / ca0) * 100.0
        total_prod = n_cb + n_cc + n_cd
        select_b = (n_cb / total_prod * 100.0) if total_prod > 1e-12 else 100.0
        heat_gen_w = (p.volume_liters * (p.delta_h1_j_per_mol * r1 + p.delta_h2_j_per_mol * r2 + p.delta_h3_j_per_mol * r3)) / 60.0
        heat_rem_w = (p.ua_j_per_min_k * (n_temp - n_temp_j)) / 60.0
        return NativeCstrSeriesParallelState(
            ca_mol_per_l=n_ca,
            cb_mol_per_l=n_cb,
            cc_mol_per_l=n_cc,
            cd_mol_per_l=n_cd,
            temp_kelvin=n_temp,
            jacket_temp_kelvin=n_temp_j,
            conversion_a_pct=conv_a,
            yield_b_pct=yield_b,
            selectivity_b_pct=select_b,
            rate_r1_mol_per_l_min=r1,
            rate_r2_mol_per_l_min=r2,
            rate_r3_mol_per_l_min=r3,
            heat_generated_watts=heat_gen_w,
            heat_removed_watts=heat_rem_w,
            is_thermal_runaway=n_temp >= threshold_k,
        )


