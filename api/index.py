import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageEnhance

app = FastAPI(title="Ultra Fast AI Enhancer Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# বিশ্বসেরা Real-ESRGAN Compact 4x3 মডেল (৪.৮ মেগাবাইট)
MODEL_PATH = "/tmp/realesr_x4v3.onnx"
PRIMARY_URL = "https://huggingface.co/CoderViking/realesr-general-x4v3-onnx/resolve/main/realesr-general-x4v3.onnx"
FALLBACK_URL = "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx"

session = None

def get_session():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            try:
                req = urllib.request.Request(PRIMARY_URL, headers=headers)
                with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                req = urllib.request.Request(FALLBACK_URL, headers=headers)
                with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                    f.write(resp.read())

        # সিপিইউ স্পিড সর্বোচ্চ বাড়ানোর জন্য Graph Optimization
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
    return session

@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
def health():
    return {"status": "online", "model": "Ultra Enhancer 4X Turbo Ready"}

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

        w, h = img.size
        
        # ১. সর্বোচ্চ গতি ও হাই-কোয়ালিটির পারফেক্ট ব্যালেন্স সাইজ (৪০০px)
        # (৪০০px ইনপুট হলে আউটপুট হবে ১৬০০x১৬০০ পিক্সেল সুপার-এইচডি)
        max_dim = 400
        if max(w, h) > max_dim:
            ratio = max_dim / float(max(w, h))
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if alpha_channel:
                alpha_channel = alpha_channel.resize((new_w, new_h), Image.Resampling.LANCZOS)

        orig_backup = img.copy()

        # ২. Real-ESRGAN এআই মডেল রান
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_tensor = np.expand_dims(img_np, axis=0)

        ort_sess = get_session()
        input_name = ort_sess.get_inputs()[0].name
        outputs = ort_sess.run(None, {input_name: img_tensor})

        out_tensor = outputs[0][0]
        out_tensor = np.clip(out_tensor, 0.0, 1.0)
        out_tensor = np.transpose(out_tensor, (1, 2, 0))
        ai_enhanced = Image.fromarray((out_tensor * 255.0).astype(np.uint8), mode="RGB")

        # ৩. ন্যাচারাল স্কিন ব্যালেন্সিং (মুখ প্লাস্টিক হবে না, আসল চামড়ার টেক্সচার থাকবে)
        orig_upscaled = orig_backup.resize(ai_enhanced.size, Image.Resampling.LANCZOS)
        enhanced_final = Image.blend(ai_enhanced, orig_upscaled, alpha=0.15)

        # ৪. স্টুডিও শার্পনেস ও আইরিস ক্ল্যারিটি ফিল্টার (চোখ, চুল ও লাইনের ডিটেইল ফুটে উঠবে)
        enhanced_final = enhanced_final.filter(ImageFilter.UnsharpMask(radius=1.4, percent=160, threshold=2))
        
        # ৫. হালকা কালার ও কনট্রাস্ট অপটিমাইজেশন
        enhanced_final = ImageEnhance.Color(enhanced_final).enhance(1.08)
        enhanced_final = ImageEnhance.Contrast(enhanced_final).enhance(1.05)

        out_io = io.BytesIO()
        if has_alpha and alpha_channel:
            alpha_resized = alpha_channel.resize(enhanced_final.size, Image.Resampling.LANCZOS)
            enhanced_final.putalpha(alpha_resized)
            enhanced_final.save(out_io, format="PNG")
            return Response(content=out_io.getvalue(), media_type="image/png")
        else:
            enhanced_final.save(out_io, format="JPEG", quality=96)
            return Response(content=out_io.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
