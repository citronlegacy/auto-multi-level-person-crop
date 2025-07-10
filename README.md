# Multi-Level Anime Person Cropper

This Python script uses pre-trained YOLO models from Hugging Face to automatically detect and crop **anime-style characters** at three levels of detail:
- Full Body (person)
- Half Body (torso)
- Head (face)

It processes entire folders of images and saves the cropped results in a structured output directory.

---

## ✨ Features

- 📥 Batch processes all `.jpg`, `.jpeg`, and `.png` images in a folder
- 🧠 Uses three YOLO models for:
  - `person` (full body)
  - `halfbody` (upper torso)
  - `head` (face with adjustable scale)
- 🧱 Outputs multiple crops per image depending on detections
- 🗂 Auto-saves output to a new folder next to your input
- 💾 CPU-only friendly (no GPU required)

---

## Requirements

- Python 3.8+
- [Pillow](https://pypi.org/project/Pillow/)
- [numpy](https://pypi.org/project/numpy/)
- [ultralytics](https://pypi.org/project/ultralytics/)
- [huggingface_hub](https://pypi.org/project/huggingface-hub/)
- [tqdm](https://pypi.org/project/tqdm/)


## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/citronlegacy/auto-multi-level-person-crop.git
   cd auto-multi-level-person-crop
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```



## Usage

1. **Run the script:**

   ```sh
   python multi-level-person-crop.py
   ```

2. **Follow the prompt:**  
-   Enter the path to your image folder (supports `.png`, `.jpg`, `.jpeg`).  
-   Cropped images will be saved in a new folder with `_output` appended.
+   When prompted, enter the full path to a folder containing your images.
+   (You can drag and drop the folder into the terminal.)
+   Cropped results will be saved in a new folder with `_output` appended to the name.


---

## Example

```
Enter folder path (or press Enter to quit): ./images
Processing folder: ./images
Output will be saved to: ./images_output
Found 20 images in ./images
Processing images: 100%|████████████████████| 20/20 [00:15<00:00,  1.33it/s]
Processed 20 images, saved 48 cropped images to ./images_output
```

---

## Model Sources

- [Anime Person Detection](https://huggingface.co/deepghs/anime_person_detection)
- [Anime Halfbody Detection](https://huggingface.co/deepghs/anime_halfbody_detection)
- [Anime Head Detection](https://huggingface.co/deepghs/anime_head_detection)

Models are downloaded automatically on first run and cached in the `./models` directory.

---

## File Overview

- `multi_level_person_crop.py` — Main script to run the cropping pipeline. It loads models, processes images, and saves crops.
- `requirements.txt` — List of required dependencies.

## Credits

- Original idea by [deepghs](https://huggingface.co/deepghs)
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
