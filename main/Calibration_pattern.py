"""
Generate holographic phase patterns for multi-plane reflection alignment.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import matplotlib.pyplot as plt
import torch

import utils as ut


#%% Basic settings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = SCRIPT_DIR / "Calibration_patterns"

SHOW_FIGURES = True
SAVE_BITMAPS = True


#%% SLM and calibration geometry

NX_SLM = 4096
NY_SLM = 2160
LAYER_NUM = 3

DX_MM = 3800e-6
DY_MM = 3800e-6

NX_LAYER = 600
SPOT_DIAMETER_MM = 2.0

SPOT_WIDTH_PX = int(SPOT_DIAMETER_MM / DX_MM)
SPOT_HEIGHT_PX = int(SPOT_DIAMETER_MM / DY_MM)

SQUARE_PHASE_VALUE = torch.pi
VORTEX_TOPOLOGICAL_CHARGE = 3

# The alignment patterns used in the experiment have no blaze carrier by
# default. Turn this on only if your optical path needs a first-order blaze.
USE_BLAZE_BACKGROUND = False
WAVELENGTH_MM = 780e-6
BLAZE_PERIOD_PIXELS = 4
BLAZE_DIRECTION = "y"


#%% Shared helper functions

def phase_to_field(phase):
    """Convert a phase map to a unit-amplitude complex field for BMP export."""
    return torch.exp(1j * torch.remainder(phase, 2 * torch.pi))


def plot_phase(phase, title, cmap="jet"):
    """Preview one phase pattern in Spyder or a normal matplotlib backend."""
    if not SHOW_FIGURES:
        return

    plt.figure(figsize=(10, 5))
    plt.imshow(torch.remainder(phase.detach().cpu(), 2 * torch.pi), cmap=cmap)
    plt.colorbar(label="phase (rad)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_phase(phase, filename):
    """Save one SLM phase pattern as a grayscale BMP."""
    if not SAVE_BITMAPS:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ut.save_phase_as_bmp(phase_to_field(phase), filename=str(OUTPUT_DIR / filename))


def write_square_region(phase, center_pixel, width_px, height_px, value):
    """Write a constant phase square centered at center_pixel=(x, y)."""
    cx, cy = map(int, center_pixel)
    half_x = int(width_px) // 2
    half_y = int(height_px) // 2

    x_start = max(cx - half_x, 0)
    x_end = min(cx + half_x, phase.shape[1])
    y_start = max(cy - half_y, 0)
    y_end = min(cy + half_y, phase.shape[0])

    phase[y_start:y_end, x_start:x_end] = value
    return phase


def make_background_phase():
    """Create the full-SLM background phase used by each calibration section."""
    if not USE_BLAZE_BACKGROUND:
        return torch.zeros((NY_SLM, NX_SLM), device=DEVICE)

    blaze = ut.BlazeOverlay(
        NY_SLM,
        NX_SLM,
        dx=DX_MM,
        wavelength=WAVELENGTH_MM,
        device=DEVICE,
    )

    zero_phase = torch.zeros((NY_SLM, NX_SLM), device=DEVICE)
    phase_total, _ = blaze.apply_blaze(
        input_pattern=zero_phase,
        period_pixels=BLAZE_PERIOD_PIXELS,
        direction=BLAZE_DIRECTION,
        show=False,
    )
    return phase_total


def make_blocks():
    """Compute the center pixel and effective area for each SLM layer block."""
    return ut.partition_slm(
        NX_SLM,
        NY_SLM,
        DX_MM,
        DY_MM,
        LAYER_NUM,
        N_eff=NX_LAYER,
        device=DEVICE,
    )


#%% Prepare reusable geometry

blocks = make_blocks()
base_phase = make_background_phase()


#%% Block x-boundary split patterns

for block_index, block in enumerate(blocks):
    x_split_phase = torch.zeros((NY_SLM, NX_SLM), device=DEVICE)
    cx, _ = block["center_pixel"]
    x_split_phase[:, : int(cx)] = torch.pi

    plot_phase(x_split_phase, f"Block {block_index} x-boundary 0-pi split")
    # save_phase(x_split_phase, f"calibration_block_{block_index}_x_split.bmp")


#%% Block y-split strip patterns

for block_index, block in enumerate(blocks):
    y_split_phase = torch.zeros((NY_SLM, NX_SLM), device=DEVICE)
    cx, cy = map(int, block["center_pixel"])
    half_x = SPOT_WIDTH_PX // 2

    x_start = max(cx - half_x, 0)
    x_end = min(cx + half_x, NX_SLM)
    y_split_phase[:cy, x_start:x_end] = torch.pi

    plot_phase(y_split_phase, f"Block {block_index} y-split strip")
    save_phase(y_split_phase, f"calibration_block_{block_index}_y_split.bmp")


#%% Block vortex calibration patterns

for block_index, block in enumerate(blocks):
    vortex_phase = torch.zeros((NY_SLM, NX_SLM), device=DEVICE)
    cx, cy = map(int, block["center_pixel"])

    half_x = SPOT_WIDTH_PX // 2
    half_y = SPOT_HEIGHT_PX // 2
    x_start = max(cx - half_x, 0)
    x_end = min(cx + half_x, NX_SLM)
    y_start = max(cy - half_y, 0)
    y_end = min(cy + half_y, NY_SLM)

    x_block = torch.arange(x_start, x_end, device=DEVICE) - cx
    y_block = torch.arange(y_start, y_end, device=DEVICE) - cy
    yy, xx = torch.meshgrid(y_block, x_block, indexing="ij")
    theta = torch.atan2(yy, xx)

    vortex_phase[y_start:y_end, x_start:x_end] = (
        VORTEX_TOPOLOGICAL_CHARGE * theta
    )

    plot_phase(
        vortex_phase,
        f"Block {block_index} vortex l={VORTEX_TOPOLOGICAL_CHARGE}",
        cmap="hsv",
    )
    save_phase(vortex_phase, f"calibration_block_{block_index}_vortex.bmp")
