import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Load YOLO model (TFLite version)
# Ultralytics automatically handles TFLite inference if tflite-runtime or tensorflow is installed
try:
    model = YOLO("notebooks/best-obj.tflite", task="detect")
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    print("Attempting to load best.pt fallback if available...")
    model = YOLO("notebooks/best.pt")

# Load MiDaS TFLite model

midas_path = "notebooks/midas_v21_small_256.tflite"
print(f"Loading MiDaS TFLite model from {midas_path}...")
interpreter = tf.lite.Interpreter(model_path=midas_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'] # [1, 256, 256, 3]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live detection... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # YOLO object detection
    results = model(frame, verbose=False)

    # MiDaS depth estimation using TFLite
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_shape[1], input_shape[2]))
    img_input = img_resized.astype(np.float32) / 255.0
    # Normalize with MiDaS specific values if needed, but often TFLite versions are simpler.
    # Standard MiDaS normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Resize depth map back to original frame size
    depth_map = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

    # Process YOLO results and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls]

            # Calculate distance using depth map in the center of the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Ensure coordinates are within image boundaries
            center_x = min(max(center_x, 0), frame.shape[1] - 1)
            center_y = min(max(center_y, 0), frame.shape[0] - 1)
            
            # Use depth map value. Note: MiDaS provides inverse depth, so higher value means closer.
            # Depth value is generally higher for closer objects.
            depth_value = depth_map[center_y, center_x]
            
            # Normalize/approximate distance
            # This constant 1000.0 is an approximation; calibration might be needed for real distances.
            distance = 1000.0 / depth_value if depth_value > 0 else 0
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            label = f"{name} {conf:.2f} Dist: {distance:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show the result
    cv2.imshow("Live TFLite Detection & Distance Estimation", frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
