# fastapi_app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from detector import detect_whitehair_bytes

app = FastAPI(title="WhiteHair REST API")

@app.post("/detect")
async def detect(file: UploadFile = File(...), thresh_l: int = 200, min_len_px: int = 10, morph_open: int = 1):
    contents = await file.read()
    res = detect_whitehair_bytes(contents, thresh_l, min_len_px, morph_open)
    payload = {
        "whitehair_ratio": res["whitehair_ratio"],
        "whitehair_pixels": res["whitehair_pixels"],
        "message": "We found some truth. Stay strong."
    }
    return JSONResponse(payload)
