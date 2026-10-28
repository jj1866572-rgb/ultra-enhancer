import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

app = FastAPI(title="Ultra Fast 4K Studio Clarity Enhancer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
def health():
    return {"status": "online", "model": "DSLR 4K High-Pass Clarity Engine Ready"}

@app.post("/")
@app.post("/api")
@app.post("/enhance")
@app.post("/api/enhance")
@app.post("/api/index.py")
@app.post("/api/index.py/enhance")
async def enhance_image(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    target = image if image is not None else file
    if target is None:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        contents = await target.read()
        raw_img = Image.open(io.BytesIO(contents))
        
        has_alpha = raw_img.mode == "RGBA"
        alpha_channel = None
        
        if has_alpha:
            alpha_channel = raw_img.split()[3]
            img = raw_img.convert("RGB")
        else:
            img = raw_img.convert("RGB")

        # আসল সাইজ অক্ষুণ্ণ রাখা (কোনো ছোট করা হবে না)
        orig_w, orig_h = img.size

        # =========================================================================
        # ১. মাল্টি-স্কেল হাই-পাস ফ্রিকোয়েন্সি এক্সট্রাকশন (Multi-Scale Texture Separation)
        # =========================================================================
        img_np = np.array(img).astype(np.float32)

        # মাইক্রো ডিটেইলস (চুলের আঁশ, চোখের পাপড়ি, স্কিন পোরস)
        fine_base = np.array(img.filter(ImageFilter.GaussianBlur(radius=1.5))).astype(np.float32)
        fine_details = img_np - fine_base

        # ম্যাক্রো ডিটেইলস (মুখের গঠন, জামার ভাঁজ ও কনট্যুর)
        macro_base = np.array(img.filter(ImageFilter.GaussianBlur(radius=4.5))).astype(np.float32)
        macro_details = img_np - macro_base

        # ডিটেইল বুস্টিং (ফটোশপ ক্ল্যারিটি টেকনিক)
        boosted_np = img_np + (fine_details * 1.5) + (macro_details * 0.7)
        boosted_np = np.clip(boosted_np, 0.0, 255.0).astype(np.uint8)
        boosted_img = Image.fromarray(boosted_np)

        # =========================================================================
        # ২. সাব-পিক্সেল আনশার্প মাস্ক (Ultra Sharp Edges)
        # =========================================================================
        sharp_img = boosted_img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=175, threshold=1))

        # =========================================================================
        # ৩. প্রফেশনাল ডেপথ ও লাইটিং টিউনিং (Rich Studio Tone)
        # =========================================================================
        final_img = ImageEnhance.Contrast(sharp_img).enhance(1.10)
        final_img = ImageEnhance.Color(final_img).enhance(1.08)
        final_img = ImageEnhance.Brightness(final_img).enhance(1.02)

        # =========================================================================
        # ৪. আউটপুট সেভ ও রিটার্ন (PNG ট্রান্সপারেন্সি অক্ষুণ্ণ রাখা)
        # =========================================================================
        out_io = io.BytesIO()
        if has_alpha and alpha_channel:
            final_img.putalpha(alpha_channel)
            final_img.save(out_io, format="PNG")
            return Response(content=out_io.getvalue(), media_type="image/png")
        else:
            final_img.save(out_io, format="JPEG", quality=98)
            return Response(content=out_io.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
