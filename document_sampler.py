import io
import json
import os
import re

import fitz  # PyMuPDF
from PIL import Image, ImageOps

    
import spacy
from layoutparser.models import EfficientDetLayoutModel
from transformers import pipeline

# ============ CONFIGURATION ============
PDF_PATH = "AdaptiveKeyframeSamplingforLongVideoUnderstanding.pdf"
OUT_DIR = "outputs_local_ai"
ANNOTATED_PDF_PATH = os.path.join(OUT_DIR, "annotated_boxes.pdf")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figures")
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Load models
print("Loading models...")
caption_cleaner = pipeline("text2text-generation", model="google/flan-t5-small")
nlp = spacy.load("en_core_web_sm")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

model = EfficientDetLayoutModel(
    "lp://PubLayNet/tf_efficientdet_d1/config",
    extra_config={"score_threshold": 0.5},
    label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
)

def detect_figures_on_page(image):
    layout = model.detect(image)
    return [b for b in layout if b.type == "figure"]


def process_figures(doc):
    results_local = []
    # Iterate over each page in the document
    for pageno in range(len(doc)):
        # Get the page from the document
        page = doc[pageno]
        # Render each page as a bitmap image with 2x scale
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        # Convert the bitmap image to a PIL image
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        # Detect figures on the page using the EfficientDet model
        figures = detect_figures_on_page(image)
        # Iterate over each detected figure
        for i, box in enumerate(figures, 1):
            # Get the coordinates of the figure
            x1, y1, x2, y2 = map(int, box.coordinates)
            # Crop the figure from the image
            fig_crop = image.crop((x1, y1, x2, y2))
            # Improve visual clarity by increasing contrast
            fig_crop = ImageOps.autocontrast(fig_crop)
            # Save the figure crop to the output directory
            fig_path = os.path.join(FIG_DIR, f"page{pageno+1}_fig{i}.png")
            fig_crop.save(fig_path)
            image_caption = ""
            try:
                # Generate a caption for the figure crop
                image_caption = captioner(fig_crop)[0]["generated_text"].strip()
            except Exception:
                # If caption generation fails, use an empty string
                image_caption = ""
            # Add the figure result to the list of results
            results_local.append({
                "type": "figure",
                "page": pageno + 1,
                "image_path": fig_path,
                "image_caption": image_caption,
            })
    return results_local


def process_embedded_images(doc):
    results_local = []
    # Iterate over each page in the document
    for pageno, page in enumerate(doc, start=1):
        # Get the image objects from the page
        imglist = page.get_images(full=True)
        # Iterate over each image
        for i, img in enumerate(imglist, start=1):
            # Extract the image from the page
            base = doc.extract_image(img[0])
            # Get the image bytes
            image_bytes = base["image"]
            # Get the image extension
            ext = base.get("ext", "png")
            # Save the image to the output directory
            img_path = os.path.join(IMG_DIR, f"page{pageno}_img{i}.{ext}")
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            caption_raw = ""
            # Get the image rectangles
            try:
                rects = page.get_image_rects(img[0])
            except Exception:
                rects = []
            if rects:
                img_rect = rects[0]
                page_rect = page.rect
                # Get the caption rectangle for the image
                caption_rect = fitz.Rect(
                    img_rect.x0, img_rect.y1, img_rect.x1, min(page_rect.y1, img_rect.y1 + 300)
                )
                try:
                    pdf_text = page.get_textbox(caption_rect).strip()
                except Exception:
                    pdf_text = ""
                if pdf_text:
                    caption_raw = pdf_text
            image_caption = ""
            try:
                # Convert the image bytes to a PIL image
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                # Generate a caption for the image using BLIP
                image_caption = captioner(pil_img)[0]["generated_text"].strip()
            except Exception:
                image_caption = ""
            results_local.append({
                "type": "embedded_image",
                "page": pageno,
                "image_path": img_path,
                "caption_raw": caption_raw,
                "image_caption": image_caption,
            })
    return results_local


def extract_algorithms(doc):
    results_local = []
    # Iterate over each page in the document
    for pageno, page in enumerate(doc, start=1):
        # Get plain text from the page
        text = page.get_text("text")
        # Use regex to find algorithm lines ex: "Algorithm 1: ADA"
        matches = re.findall(r"Algorithm\s*\d+[:\s\-]+[^\n]+", text)
        for match in matches:
            # If the algorithm contains "ADA" or "Adaptive"
            if "ADA" in match or "Adaptive" in match:
                # Add the algorithm to the results
                results_local.append({
                    "type": "algorithm",
                    "page": pageno,
                    "id": match.split(":")[0].strip(),
                    "caption": match.strip(),
                    "image_path": None,
                })
    return results_local


def apply_keywords(results):
    # Iterate over each result
    for r in results:
        caption_text = ""
        # If the result is a figure, use the image caption
        if r.get("type") == "figure":
            caption_text = r.get("image_caption", "")
        # If the result is an embedded image, use the caption raw or image caption
        elif r.get("type") == "embedded_image":
            caption_text = r.get("caption_raw") or r.get("image_caption", "")
        if caption_text:
            # Use spaCy NER to extract keywords from the caption
            doc_nlp = nlp(caption_text)
            # Add the keywords to the result
            r["keywords"] = [ent.text for ent in doc_nlp.ents]


def main():
    doc = fitz.open(PDF_PATH)
    print(f"Loaded PDF with {len(doc)} pages.")
    results = []
    results.extend(process_figures(doc))
    results.extend(process_embedded_images(doc))
    results.extend(extract_algorithms(doc))
    apply_keywords(results)
    json_path = os.path.join(OUT_DIR, "structured_output.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExtraction complete! {len(results)} items found.")
    print("Structured JSON saved to:", json_path)


if __name__ == "__main__":
    main()
