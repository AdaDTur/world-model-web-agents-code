"""
Minimal OpenAI-compatible inference server for vision-language models.
Supports Qwen2.5-VL and similar HF models.

Usage:
    conda run -n arena-env python phi4_server.py [--port 8000] [--gpu 0]
    conda run -n arena-env python phi4_server.py --model Qwen/Qwen2.5-VL-7B-Instruct --port 8000
"""

import argparse
import base64
import time
import uuid
from io import BytesIO

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import uvicorn

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

app = FastAPI()
model = None
processor = None
model_id = DEFAULT_MODEL


def load_model(gpu_id: int = 0, model_name: str = DEFAULT_MODEL, max_gpu_memory_mib: int = 0):
    global model, processor, model_id
    model_id = model_name
    print(f"[server] Loading {model_name} on cuda:{gpu_id}...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}")
    if max_gpu_memory_mib > 0:
        kwargs["max_memory"] = {gpu_id: f"{max_gpu_memory_mib}MiB"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    model.eval()
    print("[server] Model loaded and ready.")


# ── request/response schemas ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str | list

class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0

# ── helpers ───────────────────────────────────────────────────────────────────

def decode_image(image_url_dict: dict) -> Image.Image:
    url = image_url_dict.get("url", "")
    if url.startswith("data:image"):
        b64 = url.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    raise ValueError(f"Unsupported image_url format: {url[:40]}")


def messages_to_qwen(messages: list[Message]):
    """Convert OpenAI messages to Qwen2.5-VL processor input."""
    images = []
    qwen_messages = []

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            qwen_messages.append({"role": msg.role, "content": [{"type": "text", "text": content}]})
        else:
            parts = []
            for part in content:
                p = part if isinstance(part, dict) else part.__dict__
                t = p.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif t == "image_url":
                    img = decode_image(p["image_url"])
                    images.append(img)
                    parts.append({"type": "image", "image": img})
            qwen_messages.append({"role": msg.role, "content": parts})

    text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text, images


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "qwen"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    text, images = messages_to_qwen(req.messages)

    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    do_sample = req.temperature > 0.0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=do_sample,
            temperature=req.temperature if do_sample else None,
        )

    n_input = inputs["input_ids"].shape[1]
    generated = output_ids[0, n_input:]
    text_out = processor.tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text_out},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": n_input,
            "completion_tokens": len(generated),
            "total_tokens": n_input + len(generated),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_gpu_memory_mib", type=int, default=0,
                        help="Cap GPU memory in MiB (0 = no cap)")
    args = parser.parse_args()

    load_model(args.gpu, args.model, args.max_gpu_memory_mib)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
