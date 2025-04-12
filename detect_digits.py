from ultralytics import YOLO
import cv2
import os
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the trained YOLO model"""
    model_path = os.path.join('runs', 'detect', '7segment_detector', 'weights', 'best.pt')
    if os.path.exists(model_path):
        logger.info(f"Loading trained model from {model_path}")
        return YOLO(model_path)
    else:
        logger.info("Using pretrained YOLOv8n model")
        return YOLO('yolov8n.pt')

def read_yolo_labels(label_path):
    """Read YOLO format labels"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append({
                'class': int(class_id),
                'bbox': [x_center, y_center, width, height]
            })
    return labels

def convert_yolo_to_pixel(yolo_bbox, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates"""
    x_center, y_center, width, height = yolo_bbox
    
    # Convert to pixel values
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert to corner format
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    return [x1, y1, x2, y2]

def detect_digits_in_image(model, image_path, show_ground_truth=False):
    """Detect digits in an image and draw bounding boxes"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]

    # Read ground truth labels if requested
    if show_ground_truth:
        label_path = image_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        ground_truth = read_yolo_labels(label_path)
        
        # Draw ground truth boxes in blue
        for gt in ground_truth:
            bbox = convert_yolo_to_pixel(gt['bbox'], width, height)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for ground truth
            cv2.putText(image, str(gt['class']), (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Run detection
    results = model(image)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detection = {
                'class': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            }
            detections.append(detection)
    
    # Sort detections from left to right
    detections.sort(key=lambda x: x['bbox'][0])
    
    # Draw predicted boxes in green
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        label = det['class']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for predictions
        
        # Draw label with confidence
        text = f"{label}: {conf:.2f}"
        cv2.putText(image, text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Get the detected number
    number = ''.join([det['class'] for det in detections])
    
    return image, number, detections

def main():
    parser = argparse.ArgumentParser(description='Detect digits in images using YOLO')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--save', action='store_true', help='Save the annotated image')
    parser.add_argument('--show-gt', action='store_true', help='Show ground truth labels')
    args = parser.parse_args()

    try:
        # Load model
        model = load_model()
        
        # Process image
        image, number, detections = detect_digits_in_image(model, args.image_path, args.show_gt)
        
        # Print results
        print(f"\nDetected number: {number}")
        print("\nDetailed detections:")
        for det in detections:
            print(f"Digit: {det['class']}, Confidence: {det['confidence']:.2f}")
        
        # Save or display result
        if args.save:
            output_path = f"detected_{os.path.basename(args.image_path)}"
            cv2.imwrite(output_path, image)
            print(f"\nSaved annotated image to: {output_path}")
        
        # Display image
        cv2.imshow('Detected Digits', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
