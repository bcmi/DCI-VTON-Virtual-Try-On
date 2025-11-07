# server/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
from infer import load_pipeline, run_tryon

app = FastAPI()
pipe = None

@app.on_event("startup")
def _startup():
    """Load the model once when the API starts."""
    global pipe
    pipe = load_pipeline()

@app.get("/healthz")
def healthz():
    """Simple endpoint to confirm the API is alive."""
    return {"ok": True}

@app.post("/tryon")
async def tryon(person: UploadFile = File(...), garment: UploadFile = File(...)):
    """Accepts two images and returns a processed image."""
    ct_ok = {"image/jpeg", "image/png"}
    if person.content_type not in ct_ok or garment.content_type not in ct_ok:
        return JSONResponse(
            {"error": "Images must be JPEG or PNG", "code": "BAD_INPUT"},
            status_code=415,
        )

    # Read the uploaded files
    p_bytes = await person.read()
    g_bytes = await garment.read()

    try:
        # Run the AI model (placeholder for now)
        out_img = run_tryon(pipe, p_bytes, g_bytes)
        buf = io.BytesIO()
        out_img.save(buf, "PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e), "code": "MODEL_ERROR"}, status_code=500)

