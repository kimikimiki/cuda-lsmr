# FP32 benchmark record

## Environment

- Date: 2026-07-12
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, 6144 MiB
- Driver: 581.57
- CUDA Toolkit: 12.9.86
- Python: `/path/to/private/source/.venv/bin/python`
- CuPy: 14.0.1
- SciPy: 1.15.3
- NumPy: 2.2.6

## Inputs

The source directory was read only. No files under
`/path/to/private/source` were modified.

| Dataset | MAT input | Shape | nnz |
| --- | --- | ---: | ---: |
| Dataset A | `debug/Dataset A/private_input.mat` | 306136 x 6177 | 5,588,894 |
| Dataset B | `debug/Dataset B/private_input.mat` | 936063 x 23394 | 19,747,202 |

Both `A_aug` and `b_aug` were loaded as FP32. CSC indices were sorted and kept as
signed int32.

## Method

- `damp=0`, `atol=1e-6`, `btol=1e-6`, `conlim=1e8`, `maxiter=5000`.
- CuPy received an explicit zero FP32 `x0`; otherwise its compatibility path can
  promote iterative solution storage.
- Both solvers were warmed for two iterations before measurement.
- MAT I/O, CPU conversion, H2D upload, and dual-format construction were outside
  timed regions.
- RHS and output were already GPU-resident during timing.
- Each wall-time sample ended with CUDA stream synchronization.
- Seven solves were collected and the median reported.

Command:

```bash
PYTHONPATH=build/python /path/to/private/source/.venv/bin/python \
  benchmarks/benchmark_real_data.py \
  --datasets Dataset A,Dataset B --frame 0 --repeats 7 --maxiter 5000 \
  --output benchmark-results/frame0-fp32.json
```

## Results

| Dataset | cuda-lsmr samples (ms) | CuPy samples (ms) |
| --- | --- | --- |
| Dataset A | 64.63, 63.93, 60.24, 61.65, 58.90, 58.18, 58.04 | 120.38, 130.84, 135.64, 127.28, 121.41, 133.53, 120.57 |
| Dataset B | 168.59, 167.11, 167.50, 165.19, 164.16, 172.12, 165.77 | 228.49, 233.76, 226.23, 229.57, 230.81, 231.82, 223.68 |

| Dataset | cuda-lsmr median | CuPy median | Ratio | Stop | Iterations | Relative x difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset A | 60.24 ms | 127.28 ms | 0.473 | 2 / 2 | 86 / 86 | 1.84e-7 |
| Dataset B | 167.11 ms | 229.57 ms | 0.728 | 2 / 2 | 96 / 96 | 6.26e-7 |

The independently evaluated `||Ax-b||/||b||` values matched between solvers:
`0.3540266` for Dataset A and `0.2360898` for Dataset B. A non-small residual is expected
for these inconsistent least-squares systems; agreement and stop code 2 are the
relevant checks.

## Interpretation

On these matrices cuda-lsmr is faster than the requested CuPy baseline, so it
meets the "basically on par" target. The result is attributable to this specific
combination of dual sparse storage, fused vector updates, native loop dispatch,
and matrix row-length distributions. It should not be extrapolated to unrelated
GPUs or sparse patterns without measurement.
