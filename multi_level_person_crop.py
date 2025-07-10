from datetime import datetime
import os
from typing import Iterator, Optional, List, Tuple
from PIL import Image
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from tqdm import tqdm

PERSON_MODEL_REPO_ID = "deepghs/anime_person_detection"
PERSON_MODEL_SUBFOLDER = "person_detect_v1.1_m"  
PERSON_MODEL_FILENAME= "model.pt"

HALFBODY_MODEL_REPO_ID = "deepghs/anime_halfbody_detection"
HALFBODY_MODEL_SUBFOLDER = "halfbody_detect_v1.0_s"   
HALFBODY_MODEL_FILENAME= "model.pt"

HEAD_MODEL_REPO_ID = "deepghs/anime_head_detection"
HEAD_MODEL_SUBFOLDER = "head_detect_v2.0_s"   
HEAD_MODEL_FILENAME= "model.pt"

# Cache dicts to store loaded models
_loaded_models = {}
models = {
    "person": None,
    "halfbody": None,
    "head": None,
}

def download_model(repo_id: str, filename: str, subfolder: str = "", cache_dir: str = "./models") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder or None,
        cache_dir=cache_dir,
    )
    return local_path

def get_model(repo_id: str, subfolder: str, filename: str, device: str = "cpu") -> YOLO:
    key = (repo_id, subfolder, filename)
    if key not in _loaded_models:
        model_path = download_model(repo_id, filename, subfolder)
        model = YOLO(model_path)
        model.to(device)
        _loaded_models[key] = model
    return _loaded_models[key]

def load_all_models(device: str = "cpu"):
    models["person"] = get_model(PERSON_MODEL_REPO_ID, PERSON_MODEL_SUBFOLDER, PERSON_MODEL_FILENAME, device)
    models["halfbody"] = get_model(HALFBODY_MODEL_REPO_ID, HALFBODY_MODEL_SUBFOLDER, HALFBODY_MODEL_FILENAME, device)
    models["head"] = get_model(HEAD_MODEL_REPO_ID, HEAD_MODEL_SUBFOLDER, HEAD_MODEL_FILENAME, device)

def detect_objects(
    image: Image.Image,
    model: YOLO,
    conf_threshold: float,
    iou_threshold: float,
    class_filter: Optional[List[int]] = None,
    label: str = "object",
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    np_img = np.array(image)
    results = model.predict(np_img, conf=conf_threshold, iou=iou_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf.cpu())
            cls = int(box.cls.cpu())
            if conf < conf_threshold:
                continue
            if class_filter and cls not in class_filter:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x0, y0, x1, y1 = xyxy.tolist()
            detections.append(((x0, y0, x1, y1), label, conf))
    return detections

def detect_person(image: Image.Image, conf_threshold=0.3, iou_threshold=0.5, device="cpu"):
    model = models["person"]
    if model is None:
        load_all_models(device)
        model = models["person"]
    # Assuming class 0 == 'person' in this model - I dont know where this came from but AI suggested it. Simple testing shows its not needed. 
    return detect_objects(image, model, conf_threshold, iou_threshold, class_filter=[0], label="person")

def detect_halfbody(image: Image.Image, conf_threshold=0.3, iou_threshold=0.5, device="cpu"):
    model = models["halfbody"]
    if model is None:
        load_all_models(device)
        model = models["halfbody"]
    return detect_objects(image, model, conf_threshold, iou_threshold, label="halfbody")

def detect_heads(image: Image.Image, conf_threshold=0.3, iou_threshold=0.5, device="cpu"):
    model = models["head"]
    if model is None:
        load_all_models(device)
        model = models["head"]
    return detect_objects(image, model, conf_threshold, iou_threshold, label="head")

def multi_level_person_crop(
    image: Image.Image,
    head_scale: float = 1.5,
) -> Iterator[Image.Image]:

    persons = detect_person(image)
    if not persons:
        return

    for (px, _, _) in persons:
        person_image = image.crop(px)
        yield person_image, "person"

        half_detects = detect_halfbody(person_image)
        if half_detects:
            half_area, _, _ = half_detects[0]
            yield person_image.crop(half_area), "halfbody"

        head_detects = detect_heads(person_image)
        if head_detects:
            # Use the first detected head and crop square around it
            (hx0, hy0, hx1, hy1), _, _ = head_detects[0]
            cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
            width = height = max(hx1 - hx0, hy1 - hy0) * head_scale
            x0, y0 = int(max(cx - width / 2, 0)), int(max(cy - height / 2, 0))
            x1, y1 = int(min(cx + width / 2, person_image.width)), int(min(cy + height / 2, person_image.height))
            head_image = person_image.crop((x0, y0, x1, y1))
            yield head_image, "head"

def process_directory(
    input_dir: str,
    output_dir: str,
    head_scale: float = 1.5,
):
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = ('.png', '.jpg', '.jpeg')

    image_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts))
    print(f"Found {len(image_files)} images in {input_dir}")
    idx = 0
    # Loop over files with a progress bar
    for fname in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, fname)
        try:
            image = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {input_path}: {e}")
            continue

        for crop_img, detection_type in multi_level_person_crop(image, head_scale):          
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_file_name, _ = os.path.splitext(fname)
            new_filename = f"{original_file_name}_{detection_type}_{timestamp}_{idx}.png"
            out_path = os.path.join(output_dir, new_filename)
            crop_img.save(out_path)
            idx += 1  # ensure unique filenames

    print(f"Processed {len(image_files)} images, saved {idx} cropped images to {output_dir}")


def main_loop():
    while True:
        folder_path = input("\nEnter folder path (or press Enter to quit): ").strip()
        if not folder_path:
            print("Exiting the application.")
            break
        if not os.path.isdir(folder_path):
            print("Invalid directory. Please try again.")
            continue

        output_dir = f"{folder_path}_output"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing folder: {folder_path}")
        print(f"Output will be saved to: {output_dir}")

        process_directory(
            input_dir=folder_path,
            output_dir=output_dir,
        )

if __name__ == "__main__":

    print(f"Loading models... This may take a while on first run.")
    # Load all models at startup
    # Timed model loading
    import time
    start_time = time.time()
    load_all_models()
    print(f"All models loaded in {time.time() - start_time:.2f} seconds")

    main_loop()