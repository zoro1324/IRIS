import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load MiDaS model for depth estimation
midas_model_type = "MiDaS_small" # Using small for faster live inference
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    results = model(frame)

    # MiDaS depth estimation
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()

    # Process YOLO results and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = int(box.cls[0])
            name = model.names[cls]

            # Calculate distance using depth map in the center of the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Use depth map value. Note: MiDaS provides inverse depth, so higher value means closer.
            # We can approximate relative distance as C / depth_value.
            depth_value = depth_map[center_y, center_x]
            distance = 1000.0 / depth_value if depth_value > 0 else 0
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {conf:.2f} Dist: {distance:.2f}"
            
            # Draw the label with a background for better visibility
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show the result
    cv2.imshow("Live Detection & Distance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
