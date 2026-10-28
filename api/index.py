import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

app = FastAPI(title="Natural Studio HD Enhancer")

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
    return {"status": "online", "model": "Natural Studio HD Ready"}

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

        # ১. মৃদু নয়েজ রিমুভাল (ডিজিটাল গ্রেইন মুছে ত্বক ফ্রেশ করা)
        # এতে মুখের অতিরিক্ত কর্কশ ভাব দূর হবে কিন্তু আসল রূপ ঠিক থাকবে
        smooth_base = img.filter(ImageFilter.SMOOTH)
        clean_img = Image.blend(img, smooth_base, alpha=0.15)

        # ২. ব্যালান্সড স্মার্ট আনশার্প মাস্ক (Balanced Pro Clarity)
        # threshold=3 দেওয়ায় ত্বকের মসৃণ অংশ পুড়বে না, 
        # শুধু চোখের তারা, ভ্রু, চুল ও শার্টের ধারগুলো ক্রিস্টাল ক্লিয়ার হবে
        sharp_img = clean_img.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=65, threshold=3)
        )

        # ৩. অতিরিক্ত সূক্ষ্ম ডিটেইল এনহ্যান্সমেন্ট
        refined_img = ImageEnhance.Sharpness(sharp_img).enhance(1.18)

        # ৪. সফট ও ন্যাচারাল টোন টিউনিং (কোনো কালার বার্ন বা ফেটে যাওয়া হবে না)
        final_img = ImageEnhance.Contrast(refined_img).enhance(1.03)
        final_img = ImageEnhance.Color(final_img).enhance(1.02)

        # ৫. আউটপুট তৈরি (PNG ট্রান্সপারেন্সি অক্ষুণ্ণ রাখা)
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
