"""
EIT Stroke Demo Backend - Multi-step workflow
Step 1: Acquire measurements (with noise)
Step 2: Reconstruct image
Step 3: Classify stroke type
"""

import base64, io, os, re
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

DATA_DIR = "/mnt/d/Programming/EIT/brainweb_stroke_samples"

app = FastAPI()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class MeasurementRequest(BaseModel):
    filename: str  # without extension


class MeasurementResponse(BaseModel):
    filename: str
    voltage_data: list  # Flattened voltage measurements
    voltage_matrix: list  # 2D array [n_patterns, n_electrodes]
    n_patterns: int
    n_electrodes: int


class ReconstructionRequest(BaseModel):
    filename: str
    voltage_data: list  # From measurement step


class ReconstructionResponse(BaseModel):
    filename: str
    recon_png_base64: str


class ClassificationRequest(BaseModel):
    filename: str


class ClassificationResponse(BaseModel):
    filename: str
    label: str
    p_ischemic: float
    p_hemorrhagic: float


# =============================================================================
# UTILITIES
# =============================================================================

def parse_filename(name: str):
    m = re.match(
        r"subject_(\d+)_((?:3|6)layer)_sample_(\d+)_(ischemic|hemorrhagic)",
        name
    )
    if not m:
        raise ValueError("Invalid filename format")
    return m.groups()


def generate_mock_voltages(n_patterns: int, n_electrodes: int, seed: int = None):
    """
    Generate realistic-looking mock voltage measurements with noise.
    """
    if seed is not None:
        np.random.seed(seed)
    
    voltages = []
    for pattern_idx in range(n_patterns):
        pattern = np.zeros(n_electrodes)
        injection_idx = pattern_idx
        
        for i in range(n_electrodes):
            dist = min(abs(i - injection_idx), n_electrodes - abs(i - injection_idx))
            pattern[i] = 1.0 * np.exp(-dist / 3.0)
        
        pattern = pattern * 10.0  # ~10 mV range
        
        # Add noise (1% noise level)
        noise = np.random.randn(n_electrodes) * 0.01 * np.abs(pattern).mean()
        pattern += noise
        
        voltages.append(pattern.tolist())
    
    return voltages


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/measurement")
def acquire_measurement(req: MeasurementRequest):
    """
    Step 1: Acquire voltage measurements from EIT system
    """
    print(f"\n[MEASUREMENT] Acquiring data for {req.filename}")
    
    npz_path = os.path.join(DATA_DIR, req.filename + ".npz")
    if not os.path.exists(npz_path):
        raise HTTPException(404, f"Sample not found: {req.filename}")
    
    # Parse for seeding
    subject, head_model, sample_id, true_label = parse_filename(req.filename)
    seed = int(subject) * 1000 + int(sample_id)
    
    # Simulate acquisition delay
    time.sleep(0.5)
    
    # Generate mock voltage data
    n_electrodes = 16
    n_patterns = n_electrodes - 1
    
    voltages = generate_mock_voltages(
        n_patterns=n_patterns,
        n_electrodes=n_electrodes,
        seed=seed
    )
    
    voltages_array = np.array(voltages)
    voltage_flat = voltages_array.flatten().tolist()
    
    print(f"  ✓ Acquired {len(voltages)} patterns ({len(voltage_flat)} measurements)")
    
    return {
        "filename": req.filename,
        "voltage_data": voltage_flat,
        "voltage_matrix": voltages,
        "n_patterns": len(voltages),
        "n_electrodes": n_electrodes
    }


@app.post("/reconstruction")
def reconstruct_image(req: ReconstructionRequest):
    """
    Step 2: Reconstruct conductivity distribution from measurements
    """
    print(f"\n[RECONSTRUCTION] Processing {req.filename}")
    
    png_path = os.path.join(DATA_DIR, req.filename + ".png")
    if not os.path.exists(png_path):
        raise HTTPException(404, f"Reconstruction image not found: {req.filename}")
    
    # Simulate reconstruction delay
    time.sleep(1.0)
    
    # Load reconstruction image
    img = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    print(f"  ✓ Reconstruction complete")
    
    return {
        "filename": req.filename,
        "recon_png_base64": img_b64
    }


@app.post("/classification")
def classify_stroke(req: ClassificationRequest):
    """
    Step 3: Classify stroke type from reconstructed image
    """
    print(f"\n[CLASSIFICATION] Classifying {req.filename}")
    
    npz_path = os.path.join(DATA_DIR, req.filename + ".npz")
    if not os.path.exists(npz_path):
        raise HTTPException(404, f"Sample not found: {req.filename}")
    
    subject, head_model, sample_id, label = parse_filename(req.filename)
    
    # Simulate classification delay
    time.sleep(0.3)
    
    # Mock classifier confidence
    p_ischemic = 0.9 if label == "ischemic" else 0.1
    p_hemo = 1.0 - p_ischemic
    
    print(f"  ✓ Classified as: {label}")
    print(f"  ✓ Confidence: P(ischemic)={p_ischemic:.3f}, P(hemorrhagic)={p_hemo:.3f}")
    
    return {
        "filename": req.filename,
        "label": label,
        "p_ischemic": round(p_ischemic, 3),
        "p_hemorrhagic": round(p_hemo, 3)
    }


@app.get("/")
def root():
    return {"message": "EIT Stroke Demo API - Multi-step workflow"}