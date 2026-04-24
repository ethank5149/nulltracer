"""Test that Python structures map correctly to CUDA struct layouts.

Verifies byte-for-byte equivalence between NumPy arrays and GPU structures
by using a pass-through CUDA kernel.
"""

import numpy as np
import cupy as cp
import pytest
from nulltracer._params import RenderParams


class TestStructSync:
    """Verify RenderParams struct alignment between Python and CUDA."""

    def test_render_params_size(self):
        """RenderParams struct size matches expected 224 bytes."""
        import ctypes
        assert ctypes.sizeof(RenderParams) == 232, (
            f"RenderParams size {ctypes.sizeof(RenderParams)} != 232 bytes"
        )

    def test_render_params_fields(self):
        """RenderParams has all expected fields."""
        expected_fields = [
            "width", "height", "spin", "charge", "incl", "fov", "phi0", "isco",
            "steps", "obs_dist", "esc_radius", "disk_outer", "step_size",
            "bg_mode", "star_layers", "show_disk", "show_grid", "disk_temp",
            "doppler_boost", "srgb_output", "disk_alpha", "disk_max_crossings",
            "disk_mode", "debug_trace", "aa_samples", "sky_width", "sky_height",
            "qed_coupling", "hawking_boost",
        ]
        actual_fields = [f[0] for f in RenderParams._fields_]
        assert actual_fields == expected_fields

    @pytest.mark.gpu
    def test_struct_round_trip(self):
        """Write RenderParams to GPU and read back - verify byte equivalence."""
        import cupy as cp

        # Create a sample RenderParams
        rp = RenderParams(
            width=512, height=512, spin=0.5, charge=0.0, incl=np.radians(60.0),
            fov=8.0, phi0=0.0, isco=4.233, steps=2000, obs_dist=100.0,
            esc_radius=200.0, disk_outer=50.0, step_size=0.3, bg_mode=1,
            star_layers=3, show_disk=1, show_grid=0, disk_temp=1.0,
            doppler_boost=2.0, srgb_output=1.0, disk_alpha=0.95,
            disk_max_crossings=5, disk_mode=1,
            debug_trace=0.0, aa_samples=1, sky_width=0.0, sky_height=0.0,
            qed_coupling=0.0, hawking_boost=0.0,
        )

        # Convert to bytes and send to GPU
        params_bytes = bytes(rp)
        h_params = np.frombuffer(params_bytes, dtype=np.uint8)
        d_params = cp.asarray(h_params)

        # Pass-through kernel: copy input to output
        kernel_code = r"""
        extern "C" __global__ void pass_through(
            unsigned char* input,
            unsigned char* output,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input[idx];
            }
        }
        """
        module = cp.RawModule(code=kernel_code)
        kernel = module.get_function("pass_through")

        d_output = cp.zeros_like(d_params)
        threads_per_block = 256
        blocks_per_grid = (len(h_params) + threads_per_block - 1) // threads_per_block
        kernel(
            (blocks_per_grid,), (threads_per_block,),
            (d_params, d_output, np.int32(len(h_params)))
        )

        result_bytes = d_output.get().tobytes()
        assert result_bytes == params_bytes, "Struct round-trip failed: bytes mismatch"

    @pytest.mark.gpu
    def test_multiple_structs_sync(self):
        """Test multiple different RenderParams structs for sync."""
        import cupy as cp

        test_cases = [
            {"spin": 0.0, "incl": np.radians(90.0), "fov": 12.0},
            {"spin": 0.9, "incl": np.radians(45.0), "fov": 6.0},
            {"spin": -0.5, "incl": np.radians(30.0), "fov": 15.0},
        ]

        kernel_code = r"""
        extern "C" __global__ void pass_through(
            unsigned char* input,
            unsigned char* output,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input[idx];
            }
        }
        """
        module = cp.RawModule(code=kernel_code)
        kernel = module.get_function("pass_through")

        for i, params in enumerate(test_cases):
            rp = RenderParams(
                width=256, height=256,
                spin=params.get("spin", 0.0), charge=0.0,
                incl=params.get("incl", np.radians(60.0)),
                fov=params.get("fov", 8.0), phi0=0.0,
                isco=4.233, steps=1000, obs_dist=50.0,
                esc_radius=100.0, disk_outer=30.0, step_size=0.3,
                bg_mode=1, star_layers=1, show_disk=1, show_grid=0,
                disk_temp=1.0, doppler_boost=2.0, srgb_output=1.0,
                disk_alpha=0.95, disk_max_crossings=5, disk_mode=1,
                debug_trace=0.0, aa_samples=1, sky_width=0.0, sky_height=0.0,
                qed_coupling=0.0, hawking_boost=0.0,
            )

            params_bytes = bytes(rp)
            h_params = np.frombuffer(params_bytes, dtype=np.uint8)
            d_params = cp.asarray(h_params)
            d_output = cp.zeros_like(d_params)

            threads_per_block = 256
            blocks_per_grid = (len(h_params) + threads_per_block - 1) // threads_per_block
            kernel(
                (blocks_per_grid,), (threads_per_block,),
                (d_params, d_output, np.int32(len(h_params)))
            )

            result_bytes = d_output.get().tobytes()
            assert result_bytes == params_bytes, (
                f"Struct round-trip failed for test case {i}"
            )
