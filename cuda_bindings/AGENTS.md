This file describes `cuda_bindings`, the low-level CUDA host API bindings
subpackage in the `cuda-python` monorepo.

## Scope and principles

- **Role**: provide low-level, close-to-CUDA interfaces under
  `cuda.bindings.*` with broad API coverage.
- **Style**: prioritize correctness and API compatibility over convenience
  wrappers. High-level ergonomics belong in `cuda_core`, not here.
- **Cross-platform**: preserve Linux and Windows behavior unless a change is
  intentionally platform-specific.

## Package architecture

- **Public module layer**: Cython modules under `cuda/bindings/` expose user
  APIs (`driver`, `runtime`, `nvrtc`, `nvjitlink`, `nvvm`, `cufile`, etc.).
- **Internal binding layer**: `cuda/bindings/_bindings/` provides lower-level
  glue and loader helpers used by public modules.
- **Platform internals**: `cuda/bindings/_internal/` contains
  platform-specific implementation files and support code.
- **Build/codegen backend**: `build_hooks.py` drives header parsing, template
  expansion, extension configuration, and Cythonization.

## Generated-source workflow

- **Do not hand-edit generated binding files**: many files under
  `cuda/bindings/` (including `*.pyx`, `*.pxd`, `*.pyx.in`, and `*.pxd.in`)
  are generated artifacts.
- **Generated files are synchronized from another repository**: changes to these
  files in this repo are expected to be overwritten by the next sync.
- **If generated output must change**: make the change at the generation source
  and sync the updated artifacts back here, rather than patching generated files
  directly in this repo.
- **Header-driven generation**: parser behavior and required CUDA headers are
  defined in `build_hooks.py`; update those rules when introducing new symbols.
- **Platform split files**: keep `_linux.pyx` and `_windows.pyx` variants
  aligned when behavior should be equivalent.

## Testing expectations

- **Primary tests**: `pytest tests/`
- **Cython tests**:
  - build: `tests/cython/build_tests.sh` (or platform equivalent)
  - run: `pytest tests/cython/`
- **Examples**: example coverage is pytest-based under `examples/`.
- **Benchmarks**: run with `pytest --benchmark-only benchmarks/` when needed.
- **Orchestrated run**: from repo root, `scripts/run_tests.sh bindings`.

## Build and environment notes

- `CUDA_HOME` or `CUDA_PATH` must point to a valid CUDA Toolkit for source
  builds that parse headers.
- `CUDA_PYTHON_PARALLEL_LEVEL` controls build parallelism.
- `CUDA_PYTHON_PARSER_CACHING` controls parser-cache behavior during generation.
- Runtime behavior is affected by
  `CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM` and
  `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING`.

## Editing guidance

- Keep CUDA return/error semantics explicit and avoid broad fallback behavior.
- Reuse existing helper layers (`_bindings`, `_internal`, `_lib`) before adding
  new one-off utilities.
- If you add or change exported APIs, update relevant docs under
  `docs/source/module/` and tests in `tests/`.
- Prefer changes that are easy to regenerate/rebuild rather than patching
  generated output directly.

## Cython conventions (hand-written modules)

These apply to hand-written Cython modules (e.g., `cuda/bindings/_nvml.pyx`),
not to generated bindings.

- For `cdef` functions that can raise, prefer `cdef int foo(...) except? -1`
  over `cdef void foo(...) except*`. The int form is faster and Cython 3
  warns on `void + except*`.
- Keep `.pxd` and `.pyx` in lockstep. Every `noexcept` / `nogil` / `except+`
  qualifier must match on both sides.
- Hand-written `.pxd` files (e.g., `lowpp.pxd`) are private and unstable;
  prefer auto-generated headers when available.
- Use `nvmlReturn_t` for NVML calls; do not mix with `cudaError_t`.
- Mark `cdef` functions `nogil` when they do not touch Python state.
- Use `cdef inline` for header-only helpers.
- Do not write fused types by hand for the public API - they are verbose and
  defeat the type-erasure goal. Write separate helpers (e.g.,
  `HANDLE_RETURN_NVRTC` vs `HANDLE_RETURN`) instead.

## C/C++ headers and cross-platform

- In C/C++ headers, prefer `<cstddef>` / `<cstdint>` inside `__cplusplus`
  guards and `<stddef.h>` / `<stdint.h>` outside. Match the surrounding
  convention.
- For C++ free-threading correctness (Python 3.13t), static
  `if (!ptr) { ptr = new ...; }` patterns are UB. Use `std::atomic` with
  compare-exchange.
- `std::make_shared` does NOT support custom deleters; use
  `std::shared_ptr<T>(new T(...), deleter)` when a custom deleter is needed.
- POSIX-only APIs (`gmtime_r`, etc.) require Windows alternatives
  (`gmtime_s`). Do not assume POSIX without a platform branch.
