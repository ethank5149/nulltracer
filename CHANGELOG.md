# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-04-22

### Added
- Complete FastAPI server implementation (`nulltracer.server`) matching the architecture specification.
- Test suite with automated checks for ISCO, Hamiltonian conservation, and shadow boundary accuracy.
- CI configuration for automated testing.
- `CHANGELOG.md` file to track versions.
- `scripts/run_tests.sh` helper script for portability.

### Changed
- Unified the `RenderParams` structs across Python and CUDA to prevent layout drift.
- Fixed integration accuracy bugs at $p_r=0$ turning points.
- Refactored disk physics: `F_max` is now computed once per pixel instead of repetitively within integration loops.
- Consolidated `CudaRenderer` and `render_frame` code paths to eliminate duplicate rendering pipelines.
- Switched to a dictionary-based lazy module importer in `__init__.py`.
- Moved developer dependencies into the `[project.optional-dependencies]` dev block.

### Fixed
- Silently incorrect handling of classical forbidden regions ($p_r^2 \le 0$).
- A non-physical hard pole reflection in all integrators; updated `S2_EPS` regularization.
- `fov` mismatch in bloom filter calculation causing excessive bleeding.
- ISCO bisection search bracket being artificially limited to 9.0M.
