This file describes `cuda_core`, the high-level Pythonic CUDA subpackage in the
`cuda-python` monorepo.

## Scope and principles

- **Role**: provide higher-level CUDA abstractions (`Device`, `Stream`,
  `Program`, `Linker`, memory resources, graphs) on top of `cuda.bindings`.
- **API intent**: keep interfaces Pythonic while preserving explicit CUDA
  behavior and error visibility.
- **Compatibility**: changes should remain compatible with supported
  `cuda.bindings` major versions (12.x and 13.x).

## Package architecture

- **Main package**: `cuda/core/` contains most Cython modules (`*.pyx`, `*.pxd`)
  implementing runtime behaviors and public objects.
- **Subsystems**:
  - memory/resource stack: `cuda/core/_memory/`
  - system-level APIs: `cuda/core/system/`
  - compile/link path: `_program.pyx`, `_linker.pyx`, `_module.pyx`
  - execution path: `_launcher.pyx`, `_launch_config.pyx`, `_stream.pyx`
- **C++ helpers**: module-specific C++ implementations live under
  `cuda/core/_cpp/`.
- **Build backend**: `build_hooks.py` handles Cython extension setup and build
  dependency wiring.

## Build and version coupling

- `build_hooks.py` determines CUDA major version from `CUDA_CORE_BUILD_MAJOR`
  or CUDA headers (`CUDA_HOME`/`CUDA_PATH`) and uses it for build decisions.
- Source builds require CUDA headers available through `CUDA_HOME` or
  `CUDA_PATH`.
- `cuda_core` expects `cuda.bindings` to be present and version-compatible.

## Testing expectations

- **Primary tests**: `pytest tests/`
- **Cython tests**:
  - build: `tests/cython/build_tests.sh` (or platform equivalent)
  - run: `pytest tests/cython/`
- **Examples**: validate affected examples in `examples/` when changing user
  workflows or public APIs.
- **Orchestrated run**: from repo root, `scripts/run_tests.sh core`.

## Runtime/build environment notes

- Runtime env vars commonly relevant:
  - `CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`
  - `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING`
- Build env vars commonly relevant:
  - `CUDA_HOME` / `CUDA_PATH`
  - `CUDA_CORE_BUILD_MAJOR`
  - `CUDA_PYTHON_PARALLEL_LEVEL`
  - `CUDA_PYTHON_COVERAGE`

## Editing guidance

- Keep user-facing behaviors coherent with docs and examples, especially around
  stream semantics, memory ownership, and compile/link flows.
- Reuse existing shared utilities in `cuda/core/_utils/` before adding new
  helpers.
- When changing Cython signatures or cimports, verify related `.pxd` and
  call-site consistency.
- Prefer explicit error propagation over silent fallback paths.
- If you change public behavior, update tests and docs under `docs/source/`.

## Cython conventions

- For `cdef` functions that can raise (e.g., via `HANDLE_RETURN`), prefer
  `cdef int foo(...) except? -1` over `cdef void foo(...) except*`. Cython 3
  warns on `void + except*` and the int form is faster.
- Keep `.pxd` and `.pyx` declarations in lockstep. Every `cpdef` in `.pyx`
  needs a `.pxd` declaration; every `noexcept`/`nogil`/`except+` qualifier
  must match on both sides.
- When you add a new `*.pxi` to a Cython module, also add it to
  `cuda_core/MANIFEST.in`. The sdist build is not tested in CI, so this is
  silently wrong otherwise.
- In hot paths (kernel-arg dispatch, per-launch property access), avoid
  Python-object round-trips. Prefer `cimport cydriver` for CUDA types, typed
  memoryviews, `_PyLong_AsByteArray` / `PyLong_AsNativeBytes`, and list
  comprehensions over generator expressions. Do not put Python `IntEnum`
  lookups on the hot path - use `cydriver` C enums internally.
- Memoize expensive validation (driver-API checks, capability probes,
  `hasattr` walks) at construction time, not per-call. Module-level imports,
  not function-local, for capability/version probes.
- Mark `cdef` functions `nogil` when they do not touch Python state.
- Never call Python methods or mutate Python attributes inside `__dealloc__` -
  Cython requires `__dealloc__` to handle only C-level cleanup.
- Cython refcount correctness: call `Py_DECREF` (or the typed `cimport`'d
  equivalent) in `except:` branches when you incref'd before the failing call.
- `cdef str` functions implicitly allow `None` returns - declare and check
  explicitly when None is not valid.
- Do not move declarations out of `cdef extern from` blocks - it changes name
  mangling and may break ABI.
- Use the exact native type at the CUDA boundary: `cpp_bool` for `bool`,
  `intptr_t` for handles, `cydriver.CU<Type>` typedefs for driver types.
  Never substitute a generic `int32_t`.
- Do not intervene with manual C-API calls when Cython generates correct code.

## Resource lifetime and stream affinity

- `stream=None` is forbidden in `cuda.core` public APIs. Raise on a `None`
  stream. (See `_memoryview.pyx` for canonical `BufferError("stream=None is
  ambiguous...")` pattern.)
- Never hard-code stream 0. Resource cleanup paths (`close()`, unmap, etc.)
  must use the stream the resource was mapped or created on, or accept an
  explicit `stream=` argument.
- Keep strong references to the carrier object, not just the source. For
  DLPack inputs, hold the `StridedMemoryView` (or at least its metadata
  capsule); for ctypes callbacks, hold the `CFUNCTYPE` wrapper.
- When a public property's value semantics change, treat it as a breaking
  change requiring deprecation.

## API surface (cuda.core-specific)

- Public enums and type aliases live in `cuda.core.typing`, not in the feature
  module that uses them. Prefer `StrEnum`-typed `Literal[...]` strings over
  re-exporting `IntEnum`s from `cuda.bindings`. For mode parameters, prefer
  plain strings to enums.
- Do not crowd the `cuda.core` top-level namespace with descriptor types.
  Route new DLPack-style conversions through `StridedMemoryView.as_X(...)`
  (parallel to `as_mdspan`), and resource constructors through
  `Device.create_X(...)`.
- Do not expose raw CUDA handles via `__int__` or public attributes. The
  established pattern is `_handle`, reached only by internal code or tests.
  (See the "_handle incident" of PR #1660 for the cautionary tale.)
- Use `nvmlReturn_t` for NVML calls and `cudaError_t` for cudart calls - do
  not mix.
