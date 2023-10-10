import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define CORS settings
origins = ["*"]  # You can change "*" to the specific origin(s) you want to allow.

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can specify specific HTTP methods here if needed.
    allow_headers=["*"],  # You can specify specific headers here if needed.
)

# Define a function to sort and filter bounding boxes
def sort_and_filter_boxes(boxes):
    # Sort boxes based on their y-axis attribute
    sorted_boxes = sorted(boxes, key=lambda x: x["box"]["y1"])
    return sorted_boxes

def extract_text_from_boxes(boxes, image, trocr_processor, trocr_model):
    extracted_text = []
    for box in boxes:
        x1, y1, x2, y2 = (
            int(box["box"]["x1"]),
            int(box["box"]["y1"]),
            int(box["box"]["x2"]),
            int(box["box"]["y2"]),
        )
        cropped_image = image[y1:y2, x1:x2]
        cropped_pil_image = Image.fromarray(cropped_image)
        pixel_values = trocr_processor(cropped_pil_image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        extracted_text.append(generated_text)
    return extracted_text

@app.post("/extract_text")
async def extract_text(file: UploadFile):
    # Read and process the uploaded image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Load the YOLO model
    yolo_model = YOLO("model/bangla+iam.pt")

    # Load the TrOCR model
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Run inference using the YOLO model
    results = yolo_model(image, conf=0.45)

    # Convert the results to JSON format
    results = results[0].tojson()
    result = json.loads(results)

    # Sort and filter bounding boxes
    sorted_and_filtered_boxes = sort_and_filter_boxes(result)

    # Extract text from sorted and filtered bounding boxes
    extracted_text = extract_text_from_boxes(sorted_and_filtered_boxes, image, trocr_processor, trocr_model)

    return {"extracted_text": extracted_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
