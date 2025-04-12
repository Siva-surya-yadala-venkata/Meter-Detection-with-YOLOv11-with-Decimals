# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from torchvision import transforms
# from PIL import Image

# # Add CRAFT directory to Python path
# sys.path.append('./CRAFT-pytorch')

# # Import CRAFT from the correct directory
# from craft import CRAFT

# # Auto-detect device (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)  # Load model to CPU or GPU

#     # Get the absolute path of the model file
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     # Load model weights securely
#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     # Remove 'module.' prefix if present (handles DataParallel models)
#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Detect text regions using CRAFT
# def detect_text_regions(image, model):
#     # Convert OpenCV image (BGR) to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_pil = Image.fromarray(image_rgb)

#     # Preprocess Image for CRAFT
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     img = transform(image_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         y, _ = model(img)

#     # Extract text score map
#     score_text = y[0, 0, :, :].cpu().numpy()

#     # Normalize and visualize the detected text regions
#     text_region_path = "text_regions.jpg"
#     cv2.imwrite(text_region_path, (score_text * 255).astype(np.uint8))
#     print(f"Text region detection saved to {text_region_path}")

#     return score_text

# # Capture image from camera
# def capture_image():
#     cap = cv2.VideoCapture(0)  # Change to external camera if needed

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         cv2.imshow("Press 'C' to Capture, 'Q' to Quit", frame)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c'):  # Press 'C' to capture
#             capture_path = "captured_image.jpg"
#             cv2.imwrite(capture_path, frame)
#             print(f"Image saved to {capture_path}")
#             break
#         elif key == ord('q'):  # Press 'Q' to quit
#             print("Exiting without capturing image.")
#             cap.release()
#             cv2.destroyAllWindows()
#             sys.exit(1)

#     cap.release()
#     cv2.destroyAllWindows()

#     return frame

# # Perform OCR using EasyOCR
# def extract_text(image):
#     try:
#         reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() and torch.backends.cudnn.is_available())
#     except Exception as e:
#         print(f"Error initializing OCR model: {e}")
#         sys.exit(1)

#     # Convert OpenCV image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     try:
#         results = reader.readtext(gray)
#     except Exception as e:
#         print(f"Error performing OCR: {e}")
#         sys.exit(1)

#     detected_text = []
#     for (bbox, text, prob) in results:
#         detected_text.append(text)
#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         # Draw bounding box around detected text
#         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

#     # Save the image with detected text
#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     # Load CRAFT model
#     craft_model = load_craft_model()

#     # Capture image
#     image = capture_image()

#     # Detect text regions
#     text_regions = detect_text_regions(image, craft_model)

#     # Extract text using OCR
#     extracted_text = extract_text(image)

#     print("Detected Text:", extracted_text)










# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from torchvision import transforms
# from PIL import Image

# # Add CRAFT directory to Python path
# sys.path.append('./CRAFT-pytorch')

# # Import CRAFT from the correct directory
# from craft import CRAFT

# # Auto-detect device (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)  # Load model to CPU or GPU

#     # Get the absolute path of the model file
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     # Load model weights securely
#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     # Remove 'module.' prefix if present (handles DataParallel models)
#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Detect text regions using CRAFT
# def detect_text_regions(image, model):
#     # Convert OpenCV image (BGR) to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_pil = Image.fromarray(image_rgb)

#     # Preprocess Image for CRAFT
#     transform = transforms.Compose([transforms.ToTensor()])
#     img = transform(image_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         y, _ = model(img)

#     # Extract text score map
#     score_text = y[0, 0, :, :].cpu().numpy()

#     # Normalize and visualize the detected text regions
#     text_region_path = "text_regions.jpg"
#     cv2.imwrite(text_region_path, (score_text * 255).astype(np.uint8))
#     print(f"Text region detection saved to {text_region_path}")

#     return score_text

# # Capture image from camera
# def capture_image():
#     cap = cv2.VideoCapture(0)  # Change to external camera if needed

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         cv2.imshow("Press 'C' to Capture, 'Q' to Quit", frame)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c'):  # Press 'C' to capture
#             capture_path = "captured_image.jpg"
#             cv2.imwrite(capture_path, frame)
#             print(f"Image saved to {capture_path}")
#             break
#         elif key == ord('q'):  # Press 'Q' to quit
#             print("Exiting without capturing image.")
#             cap.release()
#             cv2.destroyAllWindows()
#             sys.exit(1)

#     cap.release()
#     cv2.destroyAllWindows()

#     return frame

# # Extract the Region of Interest (ROI) containing the meter digits
# def extract_roi(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur and edge detection
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         print("No contours found.")
#         return image  # Return original if no contours detected

#     # Find the largest rectangular contour
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)

#     # Expand bounding box slightly for better OCR
#     x = max(0, x - 10)
#     y = max(0, y - 10)
#     w = min(image.shape[1] - x, w + 20)
#     h = min(image.shape[0] - y, h + 20)

#     roi = image[y:y + h, x:x + w]

#     # Save the cropped image
#     cropped_path = "cropped_roi.jpg"
#     cv2.imwrite(cropped_path, roi)
#     print(f"Extracted ROI saved to {cropped_path}")

#     return roi

# # Perform OCR using EasyOCR
# def extract_text(image):
#     try:
#         reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() and torch.backends.cudnn.is_available())
#     except Exception as e:
#         print(f"Error initializing OCR model: {e}")
#         sys.exit(1)

#     # Convert OpenCV image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply OCR only on ROI
#     results = reader.readtext(gray)

#     detected_text = []
#     for (bbox, text, prob) in results:
#         detected_text.append(text)
#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         # Draw bounding box around detected text
#         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

#     # Save the image with detected text
#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     # Load CRAFT model
#     craft_model = load_craft_model()

#     # Capture image
#     image = capture_image()

#     # Detect text regions
#     text_regions = detect_text_regions(image, craft_model)

#     # Extract ROI
#     roi_image = extract_roi(image)

#     # Extract text using OCR
#     extracted_text = extract_text(roi_image)

#     print("Detected Text:", extracted_text)










# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from torchvision import transforms
# from PIL import Image

# # Add CRAFT directory to Python path
# sys.path.append('./CRAFT-pytorch')

# # Import CRAFT from the correct directory
# from craft import CRAFT

# # Auto-detect device (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)  # Load model to CPU or GPU

#     # Get the absolute path of the model file
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     # Load model weights securely
#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     # Remove 'module.' prefix if present (handles DataParallel models)
#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Capture image from camera with ROI box
# def capture_image():
#     cap = cv2.VideoCapture(0)  # Change to external camera if needed

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     # Define ROI box dimensions
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     roi_x = int(frame_width * 0.3)
#     roi_y = int(frame_height * 0.4)
#     roi_w = int(frame_width * 0.4)
#     roi_h = int(frame_height * 0.2)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         # Draw blue ROI box
#         cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

#         cv2.imshow("Press 'C' to Capture, 'Q' to Quit", frame)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c'):  # Press 'C' to capture
#             capture_path = "captured_image.jpg"
#             cv2.imwrite(capture_path, frame)
#             print(f"Image saved to {capture_path}")
#             break
#         elif key == ord('q'):  # Press 'Q' to quit
#             print("Exiting without capturing image.")
#             cap.release()
#             cv2.destroyAllWindows()
#             sys.exit(1)

#     cap.release()
#     cv2.destroyAllWindows()

#     return frame, (roi_x, roi_y, roi_w, roi_h)

# # Extract the fixed ROI containing the text
# def extract_roi(image, roi_box):
#     x, y, w, h = roi_box
#     roi = image[y:y + h, x:x + w]

#     cropped_path = "cropped_roi.jpg"
#     cv2.imwrite(cropped_path, roi)
#     print(f"Extracted ROI saved to {cropped_path}")

#     return roi

# # Perform OCR with bounding box inside ROI
# def extract_text(image, roi_box):
#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#     # Convert image to grayscale and enhance contrast
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     results = reader.readtext(gray)

#     detected_text = []
#     x_roi, y_roi, w_roi, h_roi = roi_box

#     for (bbox, text, prob) in results:
#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         # Calculate text bounding box area
#         text_x, text_y = top_left
#         text_w = bottom_right[0] - text_x
#         text_h = bottom_right[1] - text_y
#         text_area = text_w * text_h

#         # Calculate ROI area
#         roi_area = w_roi * h_roi

#         # Ensure text is inside the ROI and covers at least 50% of it
#         if (
#             x_roi <= text_x <= x_roi + w_roi and 
#             y_roi <= text_y <= y_roi + h_roi and 
#             text_area >= 0.5 * roi_area
#         ):
#             detected_text.append(text)
#             cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

#     # Save the detected text image
#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     # Load CRAFT model
#     craft_model = load_craft_model()

#     # Capture image and ROI box
#     image, roi_box = capture_image()

#     # Extract ROI
#     roi_image = extract_roi(image, roi_box)

#     # Extract text using OCR, ensuring bounding box fits inside ROI
#     extracted_text = extract_text(image, roi_box)

#     print("Detected Text:", extracted_text)









# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from craft import CRAFT

# # Auto-detect device (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)  # Load model to CPU or GPU
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     # Load model weights securely
#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Capture image from camera with ROI box
# def capture_image():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     roi_x = int(frame_width * 0.3)
#     roi_y = int(frame_height * 0.4)
#     roi_w = int(frame_width * 0.4)
#     roi_h = int(frame_height * 0.2)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         # Draw ROI box
#         cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

#         cv2.imshow("Auto Capture Mode (Press 'Q' to Quit)", frame)
        
#         # Automatically capture when a frame is clear
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     capture_path = "captured_image.jpg"
#     cv2.imwrite(capture_path, frame)
#     print(f"Image saved to {capture_path}")

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame, (roi_x, roi_y, roi_w, roi_h)

# # Extract and sharpen the ROI
# def extract_roi(image, roi_box):
#     x, y, w, h = roi_box
#     roi = image[y:y + h, x:x + w]

#     # Sharpen the image
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
#     sharpened = cv2.filter2D(roi, -1, kernel)

#     cropped_path = "cropped_roi.jpg"
#     cv2.imwrite(cropped_path, sharpened)
#     print(f"Extracted ROI saved to {cropped_path}")

#     return sharpened

# # Perform OCR with bounding box inside ROI
# def extract_text(image, roi_box):
#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#     results = reader.readtext(image)

#     detected_text = []
#     x_roi, y_roi, w_roi, h_roi = roi_box

#     for (bbox, text, prob) in results:
#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         text_x, text_y = top_left
#         text_w = bottom_right[0] - text_x
#         text_h = bottom_right[1] - text_y
#         text_area = text_w * text_h
#         roi_area = w_roi * h_roi

#         # Ensure text is inside the ROI and covers at least 50% of it
#         if (
#             x_roi <= text_x <= x_roi + w_roi and 
#             y_roi <= text_y <= y_roi + h_roi and 
#             text_area >= 0.5 * roi_area
#         ):
#             detected_text.append(text)
#             cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     craft_model = load_craft_model()
#     image, roi_box = capture_image()
#     roi_image = extract_roi(image, roi_box)
#     extracted_text = extract_text(roi_image, roi_box)

#     print("Detected Text:", extracted_text)



# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from craft import CRAFT

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Capture image with ROI box
# def capture_image():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     roi_x = int(frame_width * 0.3)
#     roi_y = int(frame_height * 0.4)
#     roi_w = int(frame_width * 0.4)
#     roi_h = int(frame_height * 0.2)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
#         cv2.imshow("Auto Capture Mode (Press 'Q' to Quit)", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     capture_path = "captured_image.jpg"
#     cv2.imwrite(capture_path, frame)
#     print(f"Image saved to {capture_path}")

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame, (roi_x, roi_y, roi_w, roi_h)

# # Extract ROI and enhance
# def extract_roi(image, roi_box):
#     x, y, w, h = roi_box
#     roi = image[y:y + h, x:x + w]

#     # Convert to grayscale
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#     # Apply adaptive thresholding
#     processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     cropped_path = "cropped_roi.jpg"
#     cv2.imwrite(cropped_path, processed)
#     print(f"Extracted ROI saved to {cropped_path}")

#     return processed

# # Perform OCR
# def extract_text(image, roi_box):
#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#     results = reader.readtext(image, detail=1)  # Keep bounding box details

#     detected_text = []
#     x_roi, y_roi, w_roi, h_roi = roi_box

#     # Debugging: Print detected text and coordinates
#     print("OCR Detected Text Boxes:")
#     for (bbox, text, prob) in results:
#         print(f"Text: '{text}', Confidence: {prob}, Bounding Box: {bbox}")

#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         # Draw bounding box for visualization
#         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
#         detected_text.append(text)

#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     craft_model = load_craft_model()
#     image, roi_box = capture_image()
#     roi_image = extract_roi(image, roi_box)
#     extracted_text = extract_text(roi_image, roi_box)

#     print("Final Detected Text:", extracted_text)



# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from craft import CRAFT

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Capture image with ROI box
# def capture_image():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     roi_x = int(frame_width * 0.3)
#     roi_y = int(frame_height * 0.4)
#     roi_w = int(frame_width * 0.4)
#     roi_h = int(frame_height * 0.2)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
#         cv2.imshow("Auto Capture Mode (Press 'Q' to Quit)", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     capture_path = "captured_image.jpg"
#     cv2.imwrite(capture_path, frame)
#     print(f"Image saved to {capture_path}")

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame, (roi_x, roi_y, roi_w, roi_h)

# # Extract ROI (No Grayscale)
# def extract_roi(image, roi_box):
#     x, y, w, h = roi_box
#     roi = image[y:y + h, x:x + w]

#     # **No Grayscale - Keeping Original Colors**
#     cropped_path = "cropped_roi.jpg"
#     cv2.imwrite(cropped_path, roi)
#     print(f"Extracted ROI saved to {cropped_path}")

#     return roi

# # Perform OCR
# def extract_text(image, roi_box):
#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#     results = reader.readtext(image, detail=1)  # Keep bounding box details

#     detected_text = []
#     x_roi, y_roi, w_roi, h_roi = roi_box

#     print("OCR Detected Text:")
#     for (bbox, text, prob) in results:
#         text = text.strip()

#         # **Fix common OCR mistakes**
#         text = text.replace('o', '0').replace('O', '0').replace('[', '').replace(']', '')

#         print(f"Text: '{text}', Confidence: {prob}, Bounding Box: {bbox}")

#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         # Draw bounding box for visualization
#         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

#         # **Ensure numeric output**
#         if text.replace('.', '', 1).isdigit():
#             detected_text.append(text)

#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     craft_model = load_craft_model()
#     image, roi_box = capture_image()
#     roi_image = extract_roi(image, roi_box)
#     extracted_text = extract_text(roi_image, roi_box)

#     print("Final Detected Text:", extracted_text)












# import sys
# import cv2
# import torch
# import numpy as np
# import easyocr
# import os
# from craft import CRAFT

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load CRAFT model
# def load_craft_model():
#     model = CRAFT().to(device)
#     model_path = os.path.abspath("C:/Users/sivas/Documents/InternShipArunSir/CRAFT-pytorch/craft_mlt_25k.pth")

#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         sys.exit(1)

#     try:
#         state_dict = torch.load(model_path, map_location=device)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         sys.exit(1)

#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

# # Capture image with automatic detection
# def capture_image():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         sys.exit(1)

#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

#     roi_x = int(frame_width * 0.3)
#     roi_y = int(frame_height * 0.4)
#     roi_w = int(frame_width * 0.4)
#     roi_h = int(frame_height * 0.2)

#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             continue

#         # Draw blue ROI box
#         cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

#         # Add scanning effect
#         scan_line_y = roi_y + (roi_h // 2) + int(10 * np.sin(cv2.getTickCount() / cv2.getTickFrequency()))
#         cv2.line(frame, (roi_x, scan_line_y), (roi_x + roi_w, scan_line_y), (0, 255, 0), 2)

#         roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

#         # OCR Detection inside ROI
#         results = reader.readtext(roi, detail=1)
#         detected_text = []
#         total_bbox_area = 0

#         for (bbox, text, prob) in results:
#             text = text.strip().replace('o', '0').replace('O', '0').replace('[', '').replace(']', '')

#             # Ensure numeric text
#             if text.replace('.', '', 1).isdigit():
#                 detected_text.append(text)

#             top_left = tuple(map(int, bbox[0]))
#             bottom_right = tuple(map(int, bbox[2]))
#             bbox_area = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
#             total_bbox_area += bbox_area

#             # Draw detected text bounding box
#             cv2.rectangle(frame, (roi_x + top_left[0], roi_y + top_left[1]), 
#                           (roi_x + bottom_right[0], roi_y + bottom_right[1]), (0, 255, 0), 2)

#         # Calculate percentage of ROI covered by text
#         roi_area = roi_w * roi_h
#         text_coverage = (total_bbox_area / roi_area) * 100

#         cv2.imshow("Scanning Mode", frame)

#         # Auto capture if text covers 50% of ROI
#         if text_coverage >= 50 and detected_text:
#             capture_path = "captured_image.jpg"
#             cv2.imwrite(capture_path, frame)
#             print(f"Image captured and saved to {capture_path}")
#             break

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting without capture.")
#             sys.exit(0)

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame, (roi_x, roi_y, roi_w, roi_h)

# # Extract ROI (No Grayscale)
# def extract_roi(image, roi_box):
#     x, y, w, h = roi_box
#     roi = image[y:y + h, x:x + w]

#     cropped_path = "cropped_roi.jpg"
#     cv2.imwrite(cropped_path, roi)
#     print(f"Extracted ROI saved to {cropped_path}")

#     return roi

# # Perform OCR with bounding box constraints
# def extract_text(image, roi_box):
#     reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#     results = reader.readtext(image, detail=1)  # Keep bounding box details

#     detected_text = []
#     x_roi, y_roi, w_roi, h_roi = roi_box

#     print("OCR Detected Text:")
#     for (bbox, text, prob) in results:
#         text = text.strip()
#         text = text.replace('o', '0').replace('O', '0').replace('[', '').replace(']', '')

#         print(f"Text: '{text}', Confidence: {prob}, Bounding Box: {bbox}")

#         top_left = tuple(map(int, bbox[0]))
#         bottom_right = tuple(map(int, bbox[2]))

#         # Ensure bounding box is inside ROI
#         if (x_roi <= top_left[0] <= x_roi + w_roi and
#             x_roi <= bottom_right[0] <= x_roi + w_roi and
#             y_roi <= top_left[1] <= y_roi + h_roi and
#             y_roi <= bottom_right[1] <= y_roi + h_roi):

#             cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
#             if text.replace('.', '', 1).isdigit():
#                 detected_text.append(text)

#     detected_text_path = "detected_text.jpg"
#     cv2.imwrite(detected_text_path, image)
#     print(f"Detected text image saved to {detected_text_path}")

#     return detected_text

# # Main function
# if __name__ == "__main__":
#     craft_model = load_craft_model()
#     image, roi_box = capture_image()
#     roi_image = extract_roi(image, roi_box)
#     extracted_text = extract_text(roi_image, roi_box)

#     print("Final Detected Text:", extracted_text)





