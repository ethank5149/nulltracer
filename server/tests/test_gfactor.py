#!/usr/bin/env python3
"""Validation test for compute_g_factor() against analytic Schwarzschild formulas.

This test compiles the compute_g_factor() CUDA function via CuPy and validates
it against exact analytic expressions for the Schwarzschild case (a=0, Q=0).
"""

import os
import sys
import math
from typing import List, Tuple

try:
    import cupy as cp
    import numpy as np
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    import numpy as np


# ──────────────────────────────────────────────────────────────────
# Analytic g-factor for Schwarzschild
# ──────────────────────────────────────────────────────────────────

def analytic_g_schwarzschild(r: float, b: float) -> float:
    """Exact g-factor for Schwarzschild (a=0, Q=0) circular orbit emitter.
    
    Formula: g = sqrt(1 - 3/r) / (1 - b * r^(-3/2))
    
    Args:
        r: Boyer-Lindquist radial coordinate in units of M
        b: Photon impact parameter (-p_phi / p_t)
    
    Returns:
        Gravitational redshift factor g
    """
    if r <= 3.0:
        return float('nan')  # No stable circular orbits for r <= 3M
    
    u_t_inv = math.sqrt(1.0 - 3.0 / r)  # 1/u^t for circular orbit
    one_minus_bOmega = 1.0 - b * r**(-1.5)  # 1 - b*Omega, where Omega = r^(-3/2)
    
    # Handle division by zero or very small denominators
    if abs(one_minus_bOmega) < 1e-30:
        return float('inf')
    
    g = u_t_inv / one_minus_bOmega
    
    # Apply same clamp as CUDA function
    g = min(max(abs(g), 0.01), 10.0)
    
    return g


# ──────────────────────────────────────────────────────────────────
# CUDA kernel source
# ──────────────────────────────────────────────────────────────────

def get_cuda_source() -> str:
    """Construct CUDA source for testing compute_g_factor().
    
    Extracts the actual implementation from disk.cu and wraps it in a test kernel.
    """
    # Read the actual compute_g_factor from disk.cu
    disk_cu_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'disk.cu')
    
    if not os.path.exists(disk_cu_path):
        raise FileNotFoundError(f"Cannot find disk.cu at {disk_cu_path}")
    
    with open(disk_cu_path, 'r') as f:
        disk_cu_content = f.read()
    
    # Extract the compute_g_factor function (lines 74-104)
    # We'll include it directly in the test kernel
    lines = disk_cu_content.split('\n')
    function_lines = []
    in_function = False
    brace_count = 0
    
    for line in lines:
        if '__device__ float compute_g_factor' in line:
            in_function = True
        
        if in_function:
            function_lines.append(line)
            # Track braces to find end of function
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and '{' in ''.join(function_lines):
                break
    
    compute_g_factor_code = '\n'.join(function_lines)
    
    # Create test kernel
    test_kernel = f"""
{compute_g_factor_code}

extern "C" __global__
void test_gfactor(const double* r_arr, const double* a_arr, 
                  const double* Q2_arr, const double* b_arr,
                  float* g_out, int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    g_out[idx] = compute_g_factor(r_arr[idx], a_arr[idx], Q2_arr[idx], b_arr[idx]);
}}
"""
    
    return test_kernel


# ──────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────

class TestCase:
    """A single test case for g-factor validation."""
    def __init__(self, name: str, r: float, a: float, Q2: float, b: float, 
                 expected: float, rel_tol: float = 1e-6):
        self.name = name
        self.r = r
        self.a = a
        self.Q2 = Q2
        self.b = b
        self.expected = expected
        self.rel_tol = rel_tol
    
    def __repr__(self):
        return f"TestCase({self.name}, r={self.r}, b={self.b})"


def get_schwarzschild_test_cases() -> List[TestCase]:
    """Generate test cases for Schwarzschild (a=0, Q=0)."""
    cases = []
    
    # Test case 1: ISCO, head-on photon
    r, b = 6.0, 0.0
    g_expected = analytic_g_schwarzschild(r, b)
    cases.append(TestCase("ISCO, head-on photon", r, 0.0, 0.0, b, g_expected))
    
    # Test case 2: ISCO, prograde photon (b = sqrt(27) ≈ 5.196 for photon sphere)
    r, b = 6.0, 5.196152422706632
    g_expected = analytic_g_schwarzschild(r, b)
    cases.append(TestCase("ISCO, prograde photon", r, 0.0, 0.0, b, g_expected))
    
    # Test case 3: r=10M, head-on
    r, b = 10.0, 0.0
    g_expected = analytic_g_schwarzschild(r, b)
    cases.append(TestCase("r=10M, head-on", r, 0.0, 0.0, b, g_expected))
    
    # Test case 4: r=10M, moderate impact parameter
    r, b = 10.0, 3.0
    g_expected = analytic_g_schwarzschild(r, b)
    cases.append(TestCase("r=10M, moderate b", r, 0.0, 0.0, b, g_expected))
    
    # Test case 5: r=20M, head-on
    r, b = 20.0, 0.0
    g_expected = analytic_g_schwarzschild(r, b)
    cases.append(TestCase("r=20M, head-on", r, 0.0, 0.0, b, g_expected))
    
    # Test case 6: Far from BH
    r, b = 100.0, 0.0
    g_expected = analytic_g_schwarzschild(r, b)
    cases.append(TestCase("r=100M, head-on", r, 0.0, 0.0, b, g_expected))
    
    return cases


def get_kerr_test_cases() -> List[TestCase]:
    """Generate sanity check test cases for Kerr (a=0.998, Q=0).
    
    For Kerr, we don't have simple analytic formulas, but we can check:
    - g > 0 for all physical radii
    - g is bounded by clamp [0.01, 10.0]
    - g → 1 as r → ∞
    """
    cases = []
    
    # Near ISCO for a=0.998 (ISCO ≈ 1.45M for prograde orbit)
    # We just check that g is positive and within bounds
    cases.append(TestCase("Kerr a=0.998, r=2M, b=0", 2.0, 0.998, 0.0, 0.0, 
                         expected=None))  # No exact expectation, just bounds check
    
    # Moderate radius
    cases.append(TestCase("Kerr a=0.998, r=10M, b=0", 10.0, 0.998, 0.0, 0.0, 
                         expected=None))
    
    # Far radius (should approach 1)
    cases.append(TestCase("Kerr a=0.998, r=100M, b=0", 100.0, 0.998, 0.0, 0.0, 
                         expected=1.0, rel_tol=0.1))  # Within 10% of 1
    
    return cases


# ──────────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    """Run all g-factor validation tests.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 70)
    print("g-factor Validation Test Suite")
    print("=" * 70)
    
    if not HAS_CUPY:
        print("\n⚠️  SKIP: CuPy not available")
        print("   Install CuPy to run GPU tests: pip install cupy-cuda11x")
        return True  # Skip is not a failure
    
    print(f"\n✓ CuPy version: {cp.__version__}")
    print(f"✓ NumPy version: {np.__version__}")
    
    # Compile CUDA kernel
    print("\n" + "─" * 70)
    print("Compiling CUDA kernel...")
    print("─" * 70)
    
    try:
        cuda_source = get_cuda_source()
        kernel = cp.RawKernel(cuda_source, 'test_gfactor')
        print("✓ Kernel compiled successfully")
    except Exception as e:
        print(f"✗ FAILED to compile kernel: {e}")
        return False
    
    # Run Schwarzschild tests
    print("\n" + "─" * 70)
    print("Schwarzschild Test Cases (a=0, Q=0)")
    print("─" * 70)
    
    schw_cases = get_schwarzschild_test_cases()
    schw_passed = 0
    schw_failed = 0
    
    for case in schw_cases:
        passed, message = run_single_test(kernel, case)
        if passed:
            schw_passed += 1
            print(f"✓ PASS: {case.name}")
            print(f"        {message}")
        else:
            schw_failed += 1
            print(f"✗ FAIL: {case.name}")
            print(f"        {message}")
    
    # Run Kerr tests (sanity checks)
    print("\n" + "─" * 70)
    print("Kerr Sanity Checks (a=0.998, Q=0)")
    print("─" * 70)
    
    kerr_cases = get_kerr_test_cases()
    kerr_passed = 0
    kerr_failed = 0
    
    for case in kerr_cases:
        passed, message = run_single_test(kernel, case, is_kerr=True)
        if passed:
            kerr_passed += 1
            print(f"✓ PASS: {case.name}")
            print(f"        {message}")
        else:
            kerr_failed += 1
            print(f"✗ FAIL: {case.name}")
            print(f"        {message}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Schwarzschild: {schw_passed}/{len(schw_cases)} passed")
    print(f"Kerr:          {kerr_passed}/{len(kerr_cases)} passed")
    
    total_passed = schw_passed + kerr_passed
    total_tests = len(schw_cases) + len(kerr_cases)
    print(f"\nTotal: {total_passed}/{total_tests} passed")
    
    all_passed = (schw_failed == 0 and kerr_failed == 0)
    if all_passed:
        print("\n✓ All tests PASSED")
    else:
        print(f"\n✗ {schw_failed + kerr_failed} tests FAILED")
    
    print("=" * 70)
    
    return all_passed


def run_single_test(kernel, case: TestCase, is_kerr: bool = False) -> Tuple[bool, str]:
    """Run a single test case.
    
    Args:
        kernel: Compiled CuPy kernel
        case: TestCase to run
        is_kerr: If True, perform only sanity checks (no exact comparison)
    
    Returns:
        (passed, message) tuple
    """
    # Prepare input arrays (single element)
    r_arr = cp.array([case.r], dtype=cp.float64)
    a_arr = cp.array([case.a], dtype=cp.float64)
    Q2_arr = cp.array([case.Q2], dtype=cp.float64)
    b_arr = cp.array([case.b], dtype=cp.float64)
    g_out = cp.zeros(1, dtype=cp.float32)
    
    # Launch kernel (1 thread)
    kernel((1,), (1,), (r_arr, a_arr, Q2_arr, b_arr, g_out, 1))
    
    # Get result
    g_computed = float(g_out[0])
    
    if is_kerr:
        # Sanity checks for Kerr
        if not (0.01 <= g_computed <= 10.0):
            return False, f"g={g_computed:.6f} outside bounds [0.01, 10.0]"
        
        if case.expected is not None:
            # Check relative tolerance
            rel_error = abs(g_computed - case.expected) / abs(case.expected)
            if rel_error > case.rel_tol:
                return False, (f"g={g_computed:.6f}, expected≈{case.expected:.6f}, "
                             f"rel_error={rel_error:.2e} > {case.rel_tol:.2e}")
            return True, f"g={g_computed:.6f}, expected≈{case.expected:.6f}, rel_error={rel_error:.2e}"
        else:
            return True, f"g={g_computed:.6f} (bounds check passed)"
    
    else:
        # Exact comparison for Schwarzschild
        expected = case.expected
        
        if math.isnan(expected) or math.isnan(g_computed):
            if math.isnan(expected) and math.isnan(g_computed):
                return True, "Both NaN (invalid orbit)"
            else:
                return False, f"g={g_computed:.6f}, expected={expected:.6f} (NaN mismatch)"
        
        if math.isinf(expected) or math.isinf(g_computed):
            # Handle infinity (denominator → 0)
            if expected > 100 and g_computed == 10.0:
                # Clamped to max
                return True, f"g={g_computed:.6f} (clamped from inf)"
            return False, f"g={g_computed:.6f}, expected={expected:.6f} (inf handling)"
        
        # Relative error
        rel_error = abs(g_computed - expected) / abs(expected)
        
        if rel_error > case.rel_tol:
            return False, (f"g={g_computed:.6f}, expected={expected:.6f}, "
                         f"rel_error={rel_error:.2e} > {case.rel_tol:.2e}")
        
        return True, f"g={g_computed:.6f}, expected={expected:.6f}, rel_error={rel_error:.2e}"


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
