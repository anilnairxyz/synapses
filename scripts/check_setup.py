"""Setup smoke test for the synapses environment.

Runs two levels of check:

1.  Always: report the environment and run a tiny tensor op on the selected
    device (CUDA on the PC, MPS on the Mac, else CPU). Fast, no downloads.

2.  --diffusers: the heavier Stable Diffusion generation test, in the spirit of
    ../PythonSnippets/generative_ai/use_diffuser.py. Needs the `diffusers` extra
    and a GPU — run this on the PC (RTX 4090), not the Mac.

Usage:
    uv run python scripts/check_setup.py
    uv run python scripts/check_setup.py --diffusers
"""

from __future__ import annotations

import argparse
import os

import torch

from synapses import device_report, get_device


def basic_check() -> torch.device:
    print("=== synapses setup check ===")
    print(device_report())

    device = get_device()
    print(f"\nRunning a 4096x4096 matmul on: {device}")
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"matmul ok — result sum = {c.sum().item():.2f}")
    print("Basic check passed.\n")
    return device


def diffusers_check(device: torch.device) -> None:
    if device.type != "cuda":
        print(
            f"[skip] --diffusers wants CUDA but selected device is '{device}'. "
            "Run this on the PC (RTX 4090)."
        )
        return

    from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    print("Model loaded successfully to CUDA.")

    prompt = "A parsons russell terrier scuba diving"
    print(f"Generating image for prompt: '{prompt}'")
    with torch.no_grad():
        image = pipe(prompt=prompt).images[0]

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/setup_check.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")
    print("Diffusers check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="synapses setup smoke test")
    parser.add_argument(
        "--diffusers",
        action="store_true",
        help="also run the Stable Diffusion generation test (needs CUDA + diffusers extra)",
    )
    args = parser.parse_args()

    device = basic_check()
    if args.diffusers:
        diffusers_check(device)

    print("Done!")


if __name__ == "__main__":
    main()
