# server/infer.py
import io
from PIL import Image
import torch

def load_pipeline():
    """Load DCI-VTON or placeholder resources."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = {"device": device}
    # later import DCI-VTON modules here and load checkpoints.
    print(f"Pipeline initialized on {device}")
    return pipe

def run_tryon(pipe, person_bytes, garment_bytes):
    """Placeholder image combine (replace with DCI-VTON inference)."""
    person = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
    # Just return the person image for now
    return person

