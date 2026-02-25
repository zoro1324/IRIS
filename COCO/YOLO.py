import cv2
import numpy as np
import tensorflow as tf
import os

# Path to the TFLite model
MODEL_PATH = r"d:\Iris\COCO\yolov8n.tflite"

# COCO Class names
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Load TFLite model and allocate tensors
    print("Loading model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape
    # Expected: [1, height, width, 3] or [1, 3, height, width]
    input_shape = input_details[0]['shape']
    input_height = input_shape[1]
    input_width = input_shape[2]
    
    print(f"Model Input Shape: {input_shape}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        h, w = frame.shape[:2]
        input_frame = cv2.resize(frame, (input_width, input_height))
        input_frame = input_frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
        input_data = np.expand_dims(input_frame, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # YOLOv8 output is typically [1, 84, 8400]
        # 84 = 4 (box coords) + 80 (class scores)
        # 8400 = total anchor points
        output = output_data[0]
        if output.shape[0] == 84:
            output = output.transpose(1, 0)  # Shape: [8400, 84]

        # Extract boxes, scores, and class IDs
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter out low-confidence detections
        threshold = 0.5
        mask = confidences > threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert boxes from [cx, cy, w, h] to [x1, y1, w, h] and rescale to frame size
        # Coordinates from the model might be normalized (0-1) or absolute (0-input_size)
        # YOLOv8 TFLite exports usually use absolute pixels relative to input size (e.g. 0-640)
        
        box_results = []
        for box in boxes:
            cx, cy, bw, bh = box
            
            # Rescale to original frame dimensions
            # If the model output is 0-640 (absolute pixels)
            x = int((cx - bw / 2) * (w / input_width))
            y = int((cy - bh / 2) * (h / input_height))
            bw_rescaled = int(bw * (w / input_width))
            bh_rescaled = int(bh * (h / input_height))
            
            box_results.append([x, y, bw_rescaled, bh_rescaled])

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(box_results, confidences.tolist(), threshold, 0.45)

        # Draw detections
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = box_results[i]
                conf = confidences[i]
                cls_id = class_ids[i]
                label = CLASSES[cls_id]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                
                # Draw label and confidence
                text = f"{label}: {conf:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("YOLOv8 Real-Time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
