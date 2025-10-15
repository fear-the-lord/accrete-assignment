# PDF Visual Extractor

Output Folder: https://drive.google.com/drive/folders/1zIdt-y-gjgqgtsi547J1BPsnPuDaqQLL?usp=sharing

Extract figures, embedded images, captions, and algorithm mentions from scientific PDFs. Outputs cropped figures, extracted embedded images, and a structured JSON summary.

## Features
- Detects figure regions using EfficientDet (PubLayNet) via LayoutParser
- Crops and saves figure images; extracts embedded images directly from the PDF
- Generates image captions using BLIP; reads nearby PDF text under images when available
- Extracts “Algorithm N: …” lines from PDF text
- Saves a single JSON with all results (figures, embedded images, algorithms)

## Repository Layout
- `document_sampler.py` — single-file script with all logic (detection, captioning, extraction)
- `outputs_local_ai/` — default output directory
  - `figures/` — cropped figures
  - `images/` — embedded images
  - `structured_output.json` — consolidated results
- Optional: `flask_app.py` — simple API (not required if you prefer single script)

## Setup
1) Python and system packages
```bash
python3 --version   # 3.11 recommended
sudo apt-get update
```

2) Python dependencies (example)
```bash
pip install pymupdf pillow layoutparser transformers spacy
python -m spacy download en_core_web_sm
```

3) (Optional) GPU acceleration for captioning
- If you have CUDA, install PyTorch with CUDA and set `device=0` when constructing the pipeline (already coded via transformers pipeline).

## Usage
Run the script on the default sample PDF:
```bash
python document_sampler.py
```
Outputs go to `outputs_local_ai/`:
- `figures/`, `images/`, `structured_output.json`

To run on your own PDF, edit `PDF_PATH` at the top of `document_sampler.py` or modify the script to accept a CLI flag (can be added on request).

## Structured Output (JSON)
Each entry is one of:
- `figure`: detected region crop
- `embedded_image`: raw embedded image from the PDF
- `algorithm`: a detected algorithm line from text

Example entry:
```json
{
  "type": "figure",
  "page": 5,
  "image_path": "outputs_local_ai/figures/page5_fig1.png",
  "image_caption": "a chart showing ...",
  "keywords": ["Dataset", "Accuracy"]
}
```

## Script Structure
The code is organized into small functions:
- `detect_figures_on_page(image)`: runs layout detection on a rendered page image and returns only figure boxes.
- `process_figures(doc)`: renders each page, crops/saves detected figures, captions each crop, returns results.
- `process_embedded_images(doc)`: extracts native embedded images per page, reads nearby PDF text below images when present, captions image content, returns results.
- `extract_algorithms(doc)`: regex finds lines like “Algorithm N: …” in page text and returns results.
- `apply_keywords(results)`: runs spaCy NER on available captions to add a `keywords` list to results.
- `main()`: orchestrates the above and writes `structured_output.json`.

## Design Document
### Approach Explored
- PDF parsing: PyMuPDF for fast page rendering, text extraction, and embedded image extraction
- Layout detection: LayoutParser’s EfficientDet (PubLayNet) to detect page elements and isolate figures
- Captioning: BLIP image captioning for semantic descriptions of figures and embedded images
- Caption text from PDF: Near-text extraction below embedded images using page coordinates
- NLP: spaCy NER to extract keywords/entities from captions
  

### Final Pipeline
1. Render each page and run EfficientDet → get figure boxes
2. Crop figure regions → caption via BLIP → save crops
3. Extract embedded images directly from the PDF XObjects → caption via BLIP
4. Try to grab nearby PDF text under each embedded image for true captions
5. Regex-detect “Algorithm N: …” lines from text
6. Run spaCy NER over available captions to produce keywords
7. Save all results to `structured_output.json`

### Why These Choices?
- EfficientDet (PubLayNet): Well-known baseline for document layout, simpler runtime than Detectron2
- BLIP captioning: Good tradeoff of quality vs speed; no OCR dependency; works well for general scientific figures
- PyMuPDF: Robust, fast PDF rendering and access to images and annotations
- spaCy small English model: Lightweight, good enough for keyword extraction

### Challenges & Mitigations
- Detectron2 dependency complexity → chose EfficientDet variant in LayoutParser to avoid heavy builds
- Caption fidelity varies with model size → made the captioner easily swappable (BLIP large or BLIP2 for better results)
- Captions vs true PDF captions → supplement with nearby PDF text beneath images to approximate ground-truth figure captions
  

### Improvements with More Time/Compute
- Switch to stronger captioners (BLIP large, BLIP2, or newer VLMs like Florence-2/Qwen-VL)
- Better figure-caption association using PDF text blocks and distance heuristics rather than a fixed 300px band
- Deduplicate similar images via perceptual hashing; cluster small icons/logos
- GPU/AMP acceleration and batching for captioning to reduce latency
- Full CLI with flags for thresholds, model choices, and page ranges
- Unit tests for extraction routines
- Dockerfile + CI workflow

## Optional: Higher-accuracy layout detector (Detectron2)
You can swap EfficientDet for Detectron2 PubLayNet models for higher accuracy (requires Detectron2 install):
- `lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config` (fast, strong)
- `lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config` (heavier)

If needed, install Detectron2 matching your PyTorch/CUDA version using their wheels. The script can be adapted to try Detectron2 first and fall back to EfficientDet.
