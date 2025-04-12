import sys
import cv2
import torch
import numpy as np
import easyocr
import os
from craft import CRAFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CRAFT model
def load_craft_model():
    model = CRAFT().to(device)
    model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Capture image with automatic detection
def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None, None

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    roi_x = int(frame_width * 0.3)
    roi_y = int(frame_height * 0.4)
    roi_w = int(frame_width * 0.4)
    roi_h = int(frame_height * 0.2)

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            continue

        # Draw ROI box
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Scanning effect
        scan_line_y = roi_y + (roi_h // 2) + int(10 * np.sin(cv2.getTickCount() / cv2.getTickFrequency()))
        cv2.line(frame, (roi_x, scan_line_y), (roi_x + roi_w, scan_line_y), (0, 255, 0), 2)

        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # OCR Detection inside ROI
        results = reader.readtext(roi, detail=1)
        detected_text = []
        total_bbox_area = 0

        for (bbox, text, prob) in results:
            text = ''.join([char for char in text if char.isalnum()])  # Character-wise detection
            if text:
                detected_text.append(text)

            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            bbox_area = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
            total_bbox_area += bbox_area

            cv2.rectangle(frame, (roi_x + top_left[0], roi_y + top_left[1]), 
                          (roi_x + bottom_right[0], roi_y + bottom_right[1]), (0, 255, 0), 2)

        # Calculate text coverage in ROI
        roi_area = roi_w * roi_h
        text_coverage = (total_bbox_area / roi_area) * 100 if roi_area > 0 else 0

        cv2.imshow("Scanning Mode", frame)

        # Auto capture if text covers at least 20% of ROI
        if text_coverage >= 20 and detected_text:
            capture_path = "captured_image.jpg"
            cv2.imwrite(capture_path, frame)
            print(f"Image captured and saved to {capture_path}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting without capture.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame if ret else None, (roi_x, roi_y, roi_w, roi_h)

# Extract ROI
def extract_roi(image, roi_box):
    if image is None:
        print("Error: No image captured.")
        return None

    x, y, w, h = roi_box
    roi = image[y:y + h, x:x + w]

    cropped_path = "cropped_roi.jpg"
    cv2.imwrite(cropped_path, roi)
    print(f"Extracted ROI saved to {cropped_path}")

    return roi

# Perform character-wise OCR on the extracted ROI
def extract_text(image, roi_box):
    if image is None:
        print("Error: No image for OCR.")
        return []

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    results = reader.readtext(image, detail=1)

    detected_text = []
    x_roi, y_roi, w_roi, h_roi = roi_box

    print("OCR Detected Characters:")
    for (bbox, text, prob) in results:
        text = ''.join([char for char in text if char.isalnum()])  # Character-wise detection
        print(f"Character: '{text}', Confidence: {prob}, Bounding Box: {bbox}")

        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Ensure bounding box is inside ROI
        if (x_roi <= top_left[0] <= x_roi + w_roi and
            x_roi <= bottom_right[0] <= x_roi + w_roi and
            y_roi <= top_left[1] <= y_roi + h_roi and
            y_roi <= bottom_right[1] <= y_roi + h_roi):

            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            if text:
                detected_text.append(text)

    detected_text_path = "detected_text.jpg"
    cv2.imwrite(detected_text_path, image)
    print(f"Detected text image saved to {detected_text_path}")

    return detected_text

# Main function
if __name__ == "__main__":
    craft_model = load_craft_model()
    if craft_model is None:
        print("CRAFT model could not be loaded. Exiting...")
        sys.exit(1)

    image, roi_box = capture_image()
    if image is None:
        print("No image captured. Exiting...")
        sys.exit(1)

    roi_image = extract_roi(image, roi_box)
    extracted_text = extract_text(roi_image, roi_box)

    print("Final Detected Characters:", extracted_text)





