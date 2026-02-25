import cv2
import numpy as np
import tensorflow as tf
import os

YOLO_PATH = r"d:\Iris\COCO\yolov8n.tflite"

CLASSES = {
  0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
  6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
  11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
  16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
  22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
  27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
  32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
  36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
  40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
  45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
  50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
  55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
  60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
  65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
  69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
  74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
  79: "toothbrush"
}

# Known average heights in centimeters (cm) for various objects
# You can tweak these based on what you want to detect
KNOWN_HEIGHTS = {
    "person": 170.0,
    "bicycle": 105.0,
    "car": 150.0,
    "motorcycle": 120.0,
    "bus": 300.0,
    "truck": 300.0,
    "cat": 30.0,
    "dog": 50.0,
    "bottle": 25.0,
    "cup": 10.0,
    "chair": 100.0,
    "laptop": 25.0,
    "cell phone": 15.0,
    "book": 20.0
}
DEFAULT_HEIGHT = 50.0 # Default fallback height

# Assumed focal length of the camera in pixels (You might need to calibrate this for your specific webcam)
# A typical value for 720p/1080p webcams is around 600 - 800
FOCAL_LENGTH = 700.0 

def estimate_distance(known_height, apparent_height_pixels):
    if apparent_height_pixels <= 0:
        return 0
    # Distance = (Known Height * Focal Length) / Apparent Height in pixels
    return (known_height * FOCAL_LENGTH) / apparent_height_pixels

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def main():
    if not os.path.exists(YOLO_PATH):
        print(f"Error: Make sure the tflite model exists at {YOLO_PATH}")
        return

    print("Loading YOLO model...")
    yolo_interpreter = load_tflite_model(YOLO_PATH)
    yolo_input_details = yolo_interpreter.get_input_details()
    yolo_output_details = yolo_interpreter.get_output_details()
    _, yolo_in_height, yolo_in_width, _ = yolo_input_details[0]['shape']
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        orig_height, orig_width = frame.shape[:2]

        # YOLOv8 preprocessing
        yolo_input_frame = cv2.resize(frame, (yolo_in_width, yolo_in_height))
        yolo_input_frame = yolo_input_frame / 255.0
        yolo_input_frame = yolo_input_frame.astype(np.float32)
        yolo_input_frame = np.expand_dims(yolo_input_frame, axis=0)

        yolo_interpreter.set_tensor(yolo_input_details[0]['index'], yolo_input_frame)
        yolo_interpreter.invoke()
        yolo_output = yolo_interpreter.get_tensor(yolo_output_details[0]['index'])
        
        # Parse YOLO output
        out = yolo_output[0]
        if out.shape[0] == 84:
            out = out.transpose(1, 0)
            
        boxes = out[:, :4]
        scores = out[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter low confidence
        score_threshold = 0.4
        valid_indices = confidences > score_threshold
        boxes = boxes[valid_indices]
        class_ids = class_ids[valid_indices]
        confidences = confidences[valid_indices]
        
        is_normalized = np.max(boxes) <= 2.0 if len(boxes) > 0 else False
        
        cv2_boxes = []
        for box, class_id in zip(boxes, class_ids):
            cx, cy, w, h = box
            
            if is_normalized:
                cx *= orig_width
                cy *= orig_height
                w *= orig_width
                h *= orig_height
            else:
                cx *= (orig_width / yolo_in_width)
                cy *= (orig_height / yolo_in_height)
                w *= (orig_width / yolo_in_width)
                h *= (orig_height / yolo_in_height)
            
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            
            cv2_boxes.append([x_min, y_min, int(w), int(h)])
            
        nms_boxes = []
        max_wh = 7680 # Avoid NMS cross-class suppression
        for j, b in enumerate(cv2_boxes):
            nms_boxes.append([b[0] + class_ids[j] * max_wh, b[1] + class_ids[j] * max_wh, b[2], b[3]])

        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), score_threshold, 0.45)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = cv2_boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]
                label = CLASSES.get(class_id, "Unknown")
                
                # Estimate distance using bounding box height
                known_h = KNOWN_HEIGHTS.get(label, DEFAULT_HEIGHT)
                distance_cm = estimate_distance(known_h, h)
                distance_m = distance_cm / 100.0

                text = f"{label} {conf:.2f} | Dist: {distance_m:.2f}m"
                
                # Draw box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Background for text
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - text_height - baseline - 5), (x + text_width, y), (0, 255, 0), -1)
                
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                print(f"Detected: {label}, Conf: {conf:.2f}, Box: [{x},{y},{w},{h}], Est. Dist: {distance_m:.2f}m")

        cv2.imshow("YOLO Box Distance Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
