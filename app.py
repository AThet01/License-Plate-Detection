import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ✅ Load your custom trained model
model = YOLO("best.pt")  # <-- Update path if different

st.title("🔍 License Plate Detection Only")
st.write("Upload an image to detect license plates.")

uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")
    img_np = np.array(image)

    # ✅ Inference with PIL image (YOLOv8 supports this)
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

            # ✅ Match based on your custom label name
            if label.lower() in ["license-plate", "plate", "lp"]:
                found_plate = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                detection_data.append({"Class": label, "Confidence": f"{conf:.2f}"})

        if found_plate:
            st.image(img_np, channels="RGB", caption="Detected License Plates")
            st.write("### Detected License Plates")
            st.table(detection_data)
        else:
            st.warning("✅ Model loaded, but no license plates were detected.")
    else:
        st.warning("✅ Model loaded, but no objects detected.")
