#!/usr/bin/env python3
"""Render hero and gallery images for Nulltracer README."""

import json
import sys
import time
import urllib.request
import urllib.error
import os

SERVER_URL = "http://localhost:8420"


def wait_for_server(timeout=30):
    """Poll /health until server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    print("Server is healthy")
                    return True
        except Exception as e:
            print(f"Waiting for server... ({e})")
            time.sleep(1)
    return False


def render(params, output_path):
    """Send render request and save image to file."""
    print(f"Rendering {output_path} with params: {params}")
    data = json.dumps(params).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/render",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}")
            img_data = resp.read()
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(img_data)
            print(f"Saved {output_path} ({len(img_data)} bytes)")
            # Check content-type header for format confirmation
            ctype = resp.headers.get("Content-Type", "")
            print(f"Content-Type: {ctype}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP Error {e.code}: {body}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


def main():
    if not wait_for_server():
        print("Server not responding", file=sys.stderr)
        sys.exit(1)

    # Base directory for images (inside container repo)
    base_dir = "/app/docs/images"
    os.makedirs(base_dir, exist_ok=True)

    configs = [
        {
            "filename": "hero.jpg",
            "params": {
                "spin": 0.94,
                "charge": 0.0,
                "inclination": 30.0,
                "fov": 10.0,
                "width": 1920,
                "height": 1080,
                "method": "kahanli8s",
                "steps": 200,
                "step_size": 0.3,
                "obs_dist": 50,
                "bg_mode": 0,
                "show_disk": True,
                "show_grid": False,
                "disk_temp": 0.6,
                "star_layers": 4,
                "srgb_output": True,
                "bloom_enabled": True,
                "bloom_radius": 1.0,
                "format": "jpeg",
                "quality": 95,
            },
        },
        {
            "filename": "gallery-schwarzschild.jpg",
            "params": {
                "spin": 0.0,
                "charge": 0.0,
                "inclination": 80.0,
                "fov": 8.0,
                "width": 1280,
                "height": 720,
                "method": "rkdp8",
                "steps": 200,
                "step_size": 0.3,
                "obs_dist": 40,
                "bg_mode": 1,
                "show_disk": True,
                "show_grid": False,
                "disk_temp": 1.0,
                "star_layers": 4,
                "srgb_output": True,
                "bloom_enabled": False,
                "format": "jpeg",
                "quality": 90,
            },
        },
        {
            "filename": "gallery-extreme-kerr.jpg",
            "params": {
                "spin": 0.998,
                "charge": 0.0,
                "inclination": 85.0,
                "fov": 12.0,
                "width": 1280,
                "height": 720,
                "method": "kahanli8s",
                "steps": 200,
                "step_size": 0.3,
                "obs_dist": 40,
                "bg_mode": 0,
                "show_disk": True,
                "show_grid": False,
                "disk_temp": 0.8,
                "star_layers": 4,
                "srgb_output": True,
                "bloom_enabled": True,
                "bloom_radius": 1.0,
                "format": "jpeg",
                "quality": 90,
            },
        },
        {
            "filename": "gallery-charged.jpg",
            "params": {
                "spin": 0.5,
                "charge": 0.7,
                "inclination": 45.0,
                "fov": 10.0,
                "width": 1280,
                "height": 720,
                "method": "rkdp8",
                "steps": 200,
                "step_size": 0.3,
                "obs_dist": 40,
                "bg_mode": 0,
                "show_disk": True,
                "show_grid": False,
                "disk_temp": 1.2,
                "star_layers": 4,
                "srgb_output": True,
                "bloom_enabled": False,
                "format": "jpeg",
                "quality": 90,
            },
        },
    ]

    for cfg in configs:
        path = os.path.join(base_dir, cfg["filename"])
        render(cfg["params"], path)
        time.sleep(1)  # Be nice to GPU between renders

    print("All renders complete.")


if __name__ == "__main__":
    main()
