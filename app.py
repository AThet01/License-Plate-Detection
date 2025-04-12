import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("best.pt") 
st.title("🔍 License Plate Detection")
st.write("Upload an image to detect license plates, process the region, and show with bounding boxes.")

uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(image)

    results = model(image)[0]
    names = model.names

    if results.boxes is not None and len(results.boxes.cls) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        detection_data = []
        found_plate = False

        for box, cls, conf in zip(boxes, clss, confs):
            label = names[int(cls)]

            if label.lower() in ["license-plate", "plate", "lp"]:
                found_plate = True
                x1, y1, x2, y2 = map(int, box)

                # Crop and preprocess the license plate
                plate_roi = img_np[y1:y2, x1:x2]
                if plate_roi.size == 0:
                    continue

                # Preprocessing (resize, grayscale, threshold)
                try:
                    plate_resized = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(plate_resized, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.adaptiveThreshold(blur, 255,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 45, 15)
                    st.image(thresh, caption="🔍 Processed License Plate", channels="GRAY")
                except:
                    thresh = plate_roi  # fallback

                # Draw bounding box on the original image
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

                detection_data.append({
                    "Class": label,
                    "Confidence": f"{conf:.2f}",
                })

        if found_plate:
            st.image(img_np, channels="RGB", caption="📷 Detected License Plates with Bounding Boxes")
            st.write("### 📄 Detection Summary")
            st.table(detection_data)
        else:
            st.warning("✅ Model loaded, but no license plates were detected.")
    else:
        st.warning("✅ Model loaded, but no objects detected.")
