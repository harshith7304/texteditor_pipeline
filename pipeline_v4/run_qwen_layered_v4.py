import os
import time
import base64
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

load_dotenv()

FAL_KEY = os.getenv("FAL_KEY")
if not FAL_KEY:
    raise RuntimeError("FAL_KEY not found. Set it in .env or environment variables.")

FAL_ENDPOINT = "https://queue.fal.run/fal-ai/qwen-image-layered"

HEADERS = {
    "Authorization": f"Key {FAL_KEY}",
    "Content-Type": "application/json"
}

INPUT_IMAGE = "image/IMAGE_CTA_BOX/Sparkling Family Joy.png"
OUTPUT_DIR = Path("outputs/IMAGE_CTA_BOX/Sparkling Family Joy_v4_t")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def image_to_base64(path: str) -> str:
    with Image.open(path) as img:
        # Resize if too large (max 1280px to be safe)
        max_size = 1280
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def download_image(url: str, save_path: Path):
    r = requests.get(url)
    r.raise_for_status()
    save_path.write_bytes(r.content)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def run_qwen_layered(image_path: str, output_dir: Path = None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Encoding image...")
    image_b64 = image_to_base64(image_path)

    # Optimized Prompting for Text/CTA Separation
    # Positive: Describes clarity and essential elements to help model focus (reduces hallucinations)
    # Negative: Standard quality guardrails
    prompt = "A high-quality professional advertisement image features different sizes of text, clear text overlays and a distinct call-to-action button"
    negative_prompt = "blurry, low quality, distortion, noise, artifacts, messy, jpeg artifacts"

    payload = {
        "image_url": f"data:image/png;base64,{image_b64}",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 50,
        "guidance_scale": 4,
        "num_layers": 2,  
        "enable_safety_checker": True,
        "output_format": "png",
        "acceleration": "regular"
    }

    print("Submitting job to fal.ai...")
    submit_resp = requests.post(FAL_ENDPOINT, headers=HEADERS, json=payload)
    
    if submit_resp.status_code != 200:
        print(f"❌ Error: API Request Failed with status {submit_resp.status_code}")
        print(f"Details: {submit_resp.text}")
    
    submit_resp.raise_for_status()
    submit_data = submit_resp.json()

    request_id = submit_data["request_id"]
    status_url = submit_data["status_url"]
    result_url = submit_data["response_url"]

    print(f"Request ID: {request_id}")
    print("Waiting for completion...")

    # Poll
    while True:
        status_resp = requests.get(status_url, headers=HEADERS)
        status_resp.raise_for_status()
        status = status_resp.json()

        if status["status"] == "COMPLETED":
            break
        elif status["status"] == "FAILED":
            raise RuntimeError("Qwen Image Layered job failed.")
        else:
            time.sleep(2)

    print("Downloading results...")
    result_resp = requests.get(result_url, headers=HEADERS)
    result_resp.raise_for_status()
    result = result_resp.json()
    
    print("DEBUG: Raw result from API:")
    print(result)

    images = result.get("images", [])
    saved_paths = []

    for idx, img_obj in enumerate(images):
        url = img_obj["url"]
        name = f"layer_{idx}"
            
        save_path = output_dir / f"{idx}_{name}.png"
        download_image(url, save_path)
        print(f"Saved: {save_path}")
        saved_paths.append(str(save_path))

    print("\n✅ Qwen Image Layered completed successfully.")
    return saved_paths

# -------------------------------------------------------
# ENTRY
# -------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        target_image = INPUT_IMAGE

    if not os.path.exists(target_image):
        print(f"Error: {target_image} not found.")
        print("Usage: python run_qwen_layered.py <path_to_image>")
    else:
        print(f"Processing: {target_image}")
        run_qwen_layered(target_image)
