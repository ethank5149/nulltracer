# Server Tests

This directory contains test suites for the nulltracer server components.

## Test Files

### `test_gfactor.py`

Validates the `compute_g_factor()` CUDA function against analytic formulas for the Schwarzschild case.

**Requirements:**
- Python 3.6+
- NumPy
- CuPy (for GPU tests; gracefully skips if not available)

**Running the test:**

```bash
# From project root
python3 server/tests/test_gfactor.py

# Or directly
./server/tests/test_gfactor.py
```

**Test Coverage:**

1. **Schwarzschild Tests (a=0, Q=0):** Six test cases comparing CUDA output against exact analytic formula:
   - ISCO (r=6M) with head-on and prograde photons
   - r=10M with head-on and moderate impact parameter
   - r=20M head-on
   - r=100M far-field case

2. **Kerr Sanity Checks (a=0.998, Q=0):** Three test cases verifying:
   - g > 0 for physical radii
   - g bounded by clamp [0.01, 10.0]
   - g → 1 as r → ∞

**Expected Output (with CuPy):**

```
======================================================================
g-factor Validation Test Suite
======================================================================

✓ CuPy version: X.X.X
✓ NumPy version: X.X.X

──────────────────────────────────────────────────────────────────────
Compiling CUDA kernel...
──────────────────────────────────────────────────────────────────────
✓ Kernel compiled successfully

──────────────────────────────────────────────────────────────────────
Schwarzschild Test Cases (a=0, Q=0)
──────────────────────────────────────────────────────────────────────
✓ PASS: ISCO, head-on photon
        g=0.707107, expected=0.707107, rel_error=X.XXe-XX
...

──────────────────────────────────────────────────────────────────────
Kerr Sanity Checks (a=0.998, Q=0)
──────────────────────────────────────────────────────────────────────
✓ PASS: Kerr a=0.998, r=2M, b=0
        g=X.XXXXXX (bounds check passed)
...

======================================================================
Test Summary
======================================================================
Schwarzschild: 6/6 passed
Kerr:          3/3 passed

Total: 9/9 passed

✓ All tests PASSED
======================================================================
```

**Without CuPy:**

```
======================================================================
g-factor Validation Test Suite
======================================================================

⚠️  SKIP: CuPy not available
   Install CuPy to run GPU tests: pip install cupy-cuda11x
```

The test exits with code 0 (success) when skipped, as missing CuPy is not a test failure.
