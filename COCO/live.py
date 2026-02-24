import cv2
import numpy as np
import tensorflow as tf
import os

YOLO_PATH = r"d:\Iris\COCO\yolov8n.tflite"
MIDAS_PATH = r"d:\Iris\COCO\midas_v21_small_256.tflite"

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

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def main():
    if not os.path.exists(YOLO_PATH) or not os.path.exists(MIDAS_PATH):
        print(f"Error: Make sure both tflite models exist at {YOLO_PATH} and {MIDAS_PATH}")
        return

    print("Loading models...")
    yolo_interpreter = load_tflite_model(YOLO_PATH)
    yolo_input_details = yolo_interpreter.get_input_details()
    yolo_output_details = yolo_interpreter.get_output_details()
    _, yolo_in_height, yolo_in_width, _ = yolo_input_details[0]['shape']

    midas_interpreter = load_tflite_model(MIDAS_PATH)
    midas_input_details = midas_interpreter.get_input_details()
    midas_output_details = midas_interpreter.get_output_details()
    _, midas_in_height, midas_in_width, _ = midas_input_details[0]['shape']
    print("Models loaded successfully.")

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        orig_height, orig_width = frame.shape[:2]

        # ==========================================
        # 1. Run YOLO Object Detection
        # ==========================================
        # YOLOv8 typical input preprocessing
        yolo_input_frame = cv2.resize(frame, (yolo_in_width, yolo_in_height))
        yolo_input_frame = yolo_input_frame / 255.0
        yolo_input_frame = yolo_input_frame.astype(np.float32)
        yolo_input_frame = np.expand_dims(yolo_input_frame, axis=0)

        yolo_interpreter.set_tensor(yolo_input_details[0]['index'], yolo_input_frame)
        yolo_interpreter.invoke()
        yolo_output = yolo_interpreter.get_tensor(yolo_output_details[0]['index'])
        
        # Parse YOLO output (handles [1, 84, 8400] or [1, 8400, 84] variants)
        out = yolo_output[0]
        if out.shape[0] == 84:
            out = out.transpose(1, 0)
            
        boxes = out[:, :4]
        scores = out[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filtering low confidence defaults
        valid_indices = confidences > 0.4
        boxes = boxes[valid_indices]
        class_ids = class_ids[valid_indices]
        confidences = confidences[valid_indices]
        
        # Check if boxes are normalized (between 0 and 1)
        is_normalized = np.max(boxes) <= 2.0 if len(boxes) > 0 else False
        
        cv2_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            
            if is_normalized:
                cx *= orig_width
                cy *= orig_height
                w *= orig_width
                h *= orig_height
            else:
                # Rescale coordinate to original frame dimensions
                cx *= (orig_width / yolo_in_width)
                cy *= (orig_height / yolo_in_height)
                w *= (orig_width / yolo_in_width)
                h *= (orig_height / yolo_in_height)
            
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            cv2_boxes.append([x_min, y_min, int(w), int(h)])
            
        # NMS to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(cv2_boxes, confidences.tolist(), 0.4, 0.4)

        # ==========================================
        # 2. Run MiDaS Depth Estimation
        # ==========================================
        # MiDaS typical input preprocessing
        midas_input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        midas_input_frame = cv2.resize(midas_input_frame, (midas_in_width, midas_in_height))
        midas_input_frame = midas_input_frame.astype(np.float32) / 255.0

        midas_mean = [0.485, 0.456, 0.406]
        midas_std = [0.229, 0.224, 0.225]
        midas_input_frame = (midas_input_frame - midas_mean) / midas_std
        midas_input_frame = np.expand_dims(midas_input_frame, axis=0).astype(np.float32)

        midas_interpreter.set_tensor(midas_input_details[0]['index'], midas_input_frame)
        midas_interpreter.invoke()
        midas_output = midas_interpreter.get_tensor(midas_output_details[0]['index'])

        depth_map = midas_output[0]
        if depth_map.ndim == 3: # In some outputs, depth has shape (H, W, 1)
            depth_map = np.squeeze(depth_map)
        depth_map = cv2.resize(depth_map, (orig_width, orig_height))
        
        # Normalize depth map for display
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = depth_map
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        
        combined_frame = frame.copy()

        # ==========================================
        # 3. Visualization and Result Merging
        # ==========================================
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = cv2_boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]
                label = CLASSES.get(class_id, "Unknown")
                
                # Estimate depth at center of bounding box
                cx = int(x + w/2)
                cy = int(y + h/2)
                cx = max(0, min(cx, orig_width - 1))
                cy = max(0, min(cy, orig_height - 1))
                rel_depth_val = depth_uint8[cy, cx]
                
                text = f"{label} {conf:.2f} (Rel Dist: {rel_depth_val})"
                
                # Plot box and text on the raw color frame
                cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(combined_frame, text, (max(0, x), max(10, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                # Also plot box and text on the colorized depth map
                cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (255, 255, 255), 3)
                cv2.putText(depth_colormap, text, (max(0, x), max(10, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)                
                
                print(f"Object: {label}, Confidence: {conf:.2f}, Box: [{x},{y},{w},{h}], Rel Dist: {rel_depth_val}")

        # Horizontally stack frame and depth map for a unified screen
        stacked_screen = np.hstack((combined_frame, depth_colormap))
        cv2.imshow("Detection and Depth Map Screens", stacked_screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
