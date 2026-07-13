#!/usr/bin/env python3
"""Benchmark cuda-lsmr and CuPy LSMR on read-only reconstruction MAT inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat


@dataclass(frozen=True)
class Timing:
    """Stores repeated solve times and their robust summary."""

    samples_ms: list[float]
    median_ms: float
    minimum_ms: float


@dataclass(frozen=True)
class DatasetResult:
    """Stores one dataset's dimensions, timing, convergence, and error metrics."""

    dataset: str
    path: str
    rows: int
    cols: int
    nnz: int
    cuda_lsmr: Timing
    cupy: Timing
    time_ratio: float
    cuda_iterations: int
    cupy_iterations: int
    cuda_stop_code: int
    cupy_stop_code: int
    relative_solution_difference: float
    cuda_relative_residual: float
    cupy_relative_residual: float
    pool_reserved_bytes: int


def parse_args() -> argparse.Namespace:
    """Parses reproducible benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--btol", type=float, default=1.0e-6)
    parser.add_argument("--conlim", type=float, default=1.0e8)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats <= 0 or args.maxiter <= 0:
        parser.error("--repeats and --maxiter must be positive")
    return args


def load_input(path: Path) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Loads A_aug and b_aug as sorted, zero-based FP32 CSC data."""
    payload = loadmat(path, variable_names=["A_aug", "b_aug"])
    if "A_aug" not in payload or "b_aug" not in payload:
        raise KeyError(f"A_aug or b_aug missing from {path}")
    matrix = sparse.csc_matrix(payload["A_aug"], dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.indices.dtype != np.int32 or matrix.indptr.dtype != np.int32:
        matrix = sparse.csc_matrix(
            (
                matrix.data,
                matrix.indices.astype(np.int32),
                matrix.indptr.astype(np.int32),
            ),
            shape=matrix.shape,
        )
    rhs = np.asarray(payload["b_aug"], dtype=np.float32).reshape(-1)
    if rhs.size != matrix.shape[0]:
        raise ValueError(f"RHS length {rhs.size} != rows {matrix.shape[0]}")
    return matrix, rhs


def measure(callable_, repeats: int) -> tuple[Timing, object]:
    """Measures synchronized solve wall time and returns the last result."""
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = callable_()
        samples.append((time.perf_counter() - started) * 1000.0)
    return (
        Timing(samples, statistics.median(samples), min(samples)),
        result,
    )


def benchmark_dataset(name: str, path: Path, args: argparse.Namespace) -> DatasetResult:
    """Runs matched FP32 warmup, timing, and residual checks for one MAT file."""
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import cupyx.scipy.sparse.linalg as cpx_linalg
    import _cuda_lsmr

    matrix, rhs = load_input(path)
    rows, cols = matrix.shape
    session = _cuda_lsmr.SparseSession(
        "csc", rows, cols, matrix.indptr, matrix.indices, matrix.data
    )
    matrix_gpu = cpx_sparse.csc_matrix(matrix)
    rhs_gpu = cp.asarray(rhs)
    custom_x = cp.empty(cols, dtype=cp.float32)
    cupy_x0 = cp.zeros(cols, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()

    def run_custom(maxiter: int = args.maxiter):
        """Runs cuda-lsmr into caller-owned CuPy storage."""
        return session.solve_device(
            int(rhs_gpu.data.ptr),
            int(custom_x.data.ptr),
            np.float32(0.0),
            np.float32(args.atol),
            np.float32(args.btol),
            np.float32(args.conlim),
            maxiter,
        )

    def run_cupy(maxiter: int = args.maxiter):
        """Runs CuPy with an explicit FP32 x0 to keep iterative state FP32."""
        result = cpx_linalg.lsmr(
            matrix_gpu,
            rhs_gpu,
            x0=cupy_x0,
            damp=np.float32(0.0),
            atol=np.float32(args.atol),
            btol=np.float32(args.btol),
            conlim=np.float32(args.conlim),
            maxiter=maxiter,
        )
        cp.cuda.get_current_stream().synchronize()
        return result

    run_custom(maxiter=2)
    run_cupy(maxiter=2)
    custom_timing, custom_result = measure(run_custom, args.repeats)
    cupy_timing, cupy_result = measure(run_cupy, args.repeats)
    assert custom_result is not None and cupy_result is not None

    cupy_x = cupy_result[0].astype(cp.float32, copy=False)
    custom_norm = float(cp.linalg.norm(custom_x).item())
    difference = float(cp.linalg.norm(custom_x - cupy_x).item())
    relative_difference = difference / max(custom_norm, 1.0e-30)
    rhs_norm = float(cp.linalg.norm(rhs_gpu).item())
    custom_residual = float(cp.linalg.norm(matrix_gpu @ custom_x - rhs_gpu).item())
    cupy_residual = float(cp.linalg.norm(matrix_gpu @ cupy_x - rhs_gpu).item())

    return DatasetResult(
        dataset=name,
        path=str(path),
        rows=rows,
        cols=cols,
        nnz=int(matrix.nnz),
        cuda_lsmr=custom_timing,
        cupy=cupy_timing,
        time_ratio=custom_timing.median_ms / cupy_timing.median_ms,
        cuda_iterations=int(custom_result.iterations),
        cupy_iterations=int(cupy_result[2]),
        cuda_stop_code=int(custom_result.stop_code),
        cupy_stop_code=int(cupy_result[1]),
        relative_solution_difference=relative_difference,
        cuda_relative_residual=custom_residual / rhs_norm,
        cupy_relative_residual=cupy_residual / rhs_norm,
        pool_reserved_bytes=int(session.pool_reserved_bytes),
    )


def print_result(result: DatasetResult) -> None:
    """Prints one compact, machine-readable benchmark summary."""
    print(
        f"[{result.dataset}] shape={result.rows}x{result.cols} nnz={result.nnz} "
        f"cuda_lsmr={result.cuda_lsmr.median_ms:.3f}ms "
        f"cupy={result.cupy.median_ms:.3f}ms ratio={result.time_ratio:.3f} "
        f"itn={result.cuda_iterations}/{result.cupy_iterations} "
        f"rel_x_diff={result.relative_solution_difference:.3e} "
        f"rel_r={result.cuda_relative_residual:.3e}/{result.cupy_relative_residual:.3e}"
    )


def main() -> None:
    """Benchmarks requested datasets and optionally saves JSON evidence."""
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    results: list[DatasetResult] = []
    for dataset in datasets:
        path = args.data_root / dataset / (
            f"frame_{args.frame:06d}_fixed_augmented_lsmr_input.mat"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        result = benchmark_dataset(dataset, path, args)
        results.append(result)
        print_result(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
