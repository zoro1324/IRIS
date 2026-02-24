import 'dart:typed_data';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/detection_result.dart';
import 'dart:ui';

/// Service for running TFLite inference (YOLO + MiDaS)
class TfliteService {
  static final TfliteService _instance = TfliteService._internal();
  factory TfliteService() => _instance;
  TfliteService._internal();

  Interpreter? _yoloInterpreter;
  Interpreter? _midasInterpreter;
  bool _isInitialized = false;

  /// YOLO model class names matching data.yaml
  static const List<String> classNames = [
    'Bicycle',
    'Bus',
    'Car',
    'Chair',
    'Cow',
    'Dogs',
    'Motorcycle',
    'Person',
    'Stair',
    'Table',
    'Trash',
    'Truck',
  ];

  /// YOLO input size (typical for YOLO TFLite)
  static const int yoloInputSize = 320;

  /// MiDaS input size
  static const int midasInputSize = 256;

  /// Confidence threshold
  static const double confidenceThreshold = 0.4;

  /// NMS IoU threshold
  static const double iouThreshold = 0.45;

  bool get isInitialized => _isInitialized;

  /// Initialize both models
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load YOLO model
      _yoloInterpreter = await Interpreter.fromAsset(
        'best-obj.tflite',
        options: InterpreterOptions()..threads = 4,
      );
      print('YOLO TFLite model loaded successfully');

      // Load MiDaS model
      _midasInterpreter = await Interpreter.fromAsset(
        'midas_v21_small_256.tflite',
        options: InterpreterOptions()..threads = 2,
      );
      print('MiDaS TFLite model loaded successfully');

      _isInitialized = true;
    } catch (e) {
      print('Error loading TFLite models: $e');
      rethrow;
    }
  }

  /// Run detection on a camera image
  Future<List<DetectionResult>> detectObjects(
    CameraImage cameraImage,
    int imageWidth,
    int imageHeight,
  ) async {
    if (!_isInitialized) return [];

    try {
      // Convert CameraImage to RGB bytes
      final rgbBytes = _convertCameraImage(cameraImage);
      if (rgbBytes == null) return [];

      // Run YOLO detection
      final detections = _runYoloInference(rgbBytes, imageWidth, imageHeight);

      // Run MiDaS depth estimation
      final depthMap = _runMidasInference(rgbBytes, imageWidth, imageHeight);

      // Combine detections with depth
      final results = <DetectionResult>[];
      for (final det in detections) {
        final centerX = ((det['x1'] + det['x2']) / 2)
            .clamp(0, imageWidth - 1)
            .toInt();
        final centerY = ((det['y1'] + det['y2']) / 2)
            .clamp(0, imageHeight - 1)
            .toInt();

        // Sample depth at center of bounding box
        final depthX = (centerX * midasInputSize / imageWidth).toInt().clamp(
          0,
          midasInputSize - 1,
        );
        final depthY = (centerY * midasInputSize / imageHeight).toInt().clamp(
          0,
          midasInputSize - 1,
        );
        final depthValue = depthMap[depthY * midasInputSize + depthX];

        // Convert inverse depth to approximate distance
        final distance = depthValue > 0 ? 1000.0 / depthValue : 999.0;

        results.add(
          DetectionResult(
            className: det['name'] as String,
            confidence: det['conf'] as double,
            boundingBox: Rect.fromLTRB(
              (det['x1'] as double) / imageWidth,
              (det['y1'] as double) / imageHeight,
              (det['x2'] as double) / imageWidth,
              (det['y2'] as double) / imageHeight,
            ),
            depthValue: distance,
            distanceCategory: DetectionResult.categorizeDepth(distance),
          ),
        );
      }

      // Sort by priority (closest first)
      results.sort((a, b) => a.priority.compareTo(b.priority));
      return results;
    } catch (e) {
      print('Detection error: $e');
      return [];
    }
  }

  /// Convert CameraImage (YUV420) to RGB Float32List for model input
  Uint8List? _convertCameraImage(CameraImage image) {
    try {
      final int width = image.width;
      final int height = image.height;
      final rgbBytes = Uint8List(width * height * 3);

      // Handle YUV420 format (most common on Android)
      if (image.format.group == ImageFormatGroup.yuv420) {
        final yPlane = image.planes[0];
        final uPlane = image.planes[1];
        final vPlane = image.planes[2];

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final yIndex = y * yPlane.bytesPerRow + x;
            final uvIndex = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2);

            final yVal = yPlane.bytes[yIndex];
            final uVal = uPlane.bytes[uvIndex];
            final vVal = vPlane.bytes[uvIndex];

            final r = (yVal + 1.370705 * (vVal - 128)).clamp(0, 255).toInt();
            final g = (yVal - 0.337633 * (uVal - 128) - 0.698001 * (vVal - 128))
                .clamp(0, 255)
                .toInt();
            final b = (yVal + 1.732446 * (uVal - 128)).clamp(0, 255).toInt();

            final idx = (y * width + x) * 3;
            rgbBytes[idx] = r;
            rgbBytes[idx + 1] = g;
            rgbBytes[idx + 2] = b;
          }
        }
      } else if (image.format.group == ImageFormatGroup.bgra8888) {
        final bytes = image.planes[0].bytes;
        for (int i = 0, j = 0; i < bytes.length; i += 4, j += 3) {
          rgbBytes[j] = bytes[i + 2]; // R
          rgbBytes[j + 1] = bytes[i + 1]; // G
          rgbBytes[j + 2] = bytes[i]; // B
        }
      }

      return rgbBytes;
    } catch (e) {
      print('Image conversion error: $e');
      return null;
    }
  }

  /// Run YOLO inference and return raw detections
  List<Map<String, dynamic>> _runYoloInference(
    Uint8List rgbBytes,
    int imageWidth,
    int imageHeight,
  ) {
    if (_yoloInterpreter == null) return [];

    // Get input/output shapes
    final inputShape = _yoloInterpreter!.getInputTensor(0).shape;
    final inputH = inputShape[1];
    final inputW = inputShape[2];

    // Preprocess: resize + normalize to [0,1]
    final input = Float32List(1 * inputH * inputW * 3);
    for (int y = 0; y < inputH; y++) {
      for (int x = 0; x < inputW; x++) {
        final srcX = (x * imageWidth / inputW).toInt().clamp(0, imageWidth - 1);
        final srcY = (y * imageHeight / inputH).toInt().clamp(
          0,
          imageHeight - 1,
        );
        final srcIdx = (srcY * imageWidth + srcX) * 3;
        final dstIdx = (y * inputW + x) * 3;

        input[dstIdx] = rgbBytes[srcIdx] / 255.0;
        input[dstIdx + 1] = rgbBytes[srcIdx + 1] / 255.0;
        input[dstIdx + 2] = rgbBytes[srcIdx + 2] / 255.0;
      }
    }

    // Reshape for model input [1, H, W, 3]
    final inputTensor = input.reshape([1, inputH, inputW, 3]);

    // Get output shape and allocate output
    final outputShape = _yoloInterpreter!.getOutputTensor(0).shape;
    // YOLO output is typically [1, num_boxes, 4+num_classes] or [1, 4+num_classes, num_boxes]
    final outputSize = outputShape.reduce((a, b) => a * b);
    final outputFlat = Float32List(outputSize);

    // Run inference
    final outputMap = <int, Object>{};
    if (outputShape.length == 3) {
      final output = outputFlat.reshape(outputShape);
      outputMap[0] = output;
    } else {
      outputMap[0] = outputFlat;
    }
    _yoloInterpreter!.runForMultipleInputs([inputTensor], outputMap);

    // Parse YOLO output
    return _parseYoloOutput(
      outputFlat,
      outputShape,
      imageWidth,
      imageHeight,
      inputW,
      inputH,
    );
  }

  /// Parse YOLO TFLite output
  List<Map<String, dynamic>> _parseYoloOutput(
    Float32List output,
    List<int> shape,
    int imageWidth,
    int imageHeight,
    int inputW,
    int inputH,
  ) {
    final detections = <Map<String, dynamic>>[];

    if (shape.length != 3) return detections;

    final dim1 = shape[1];
    final dim2 = shape[2];

    // Determine format: [1, num_detections, 4+nc] or [1, 4+nc, num_detections]
    final numClasses = classNames.length;
    bool transposed = false;
    int numDetections;

    if (dim2 == 4 + numClasses) {
      // Standard format: [1, num_detections, 4 + numClasses]
      numDetections = dim1;
    } else if (dim1 == 4 + numClasses) {
      // Transposed format: [1, 4 + numClasses, num_detections]
      transposed = true;
      numDetections = dim2;
    } else {
      // Try to figure out which dim is the detection dim
      // Assume the larger dimension is num_detections
      if (dim1 > dim2) {
        numDetections = dim1;
      } else {
        transposed = true;
        numDetections = dim2;
      }
    }

    for (int i = 0; i < numDetections; i++) {
      double cx, cy, w, h;
      List<double> classScores = [];

      if (transposed) {
        cx = output[0 * numDetections + i];
        cy = output[1 * numDetections + i];
        w = output[2 * numDetections + i];
        h = output[3 * numDetections + i];
        for (int c = 0; c < numClasses; c++) {
          classScores.add(output[(4 + c) * numDetections + i]);
        }
      } else {
        final stride = 4 + numClasses;
        cx = output[i * stride + 0];
        cy = output[i * stride + 1];
        w = output[i * stride + 2];
        h = output[i * stride + 3];
        for (int c = 0; c < numClasses; c++) {
          classScores.add(output[i * stride + 4 + c]);
        }
      }

      // Find max class score
      double maxScore = -1;
      int maxIdx = 0;
      for (int c = 0; c < classScores.length; c++) {
        if (classScores[c] > maxScore) {
          maxScore = classScores[c];
          maxIdx = c;
        }
      }

      if (maxScore < confidenceThreshold) continue;

      // Convert from center format to corner format, scale to image size
      final scaleX = imageWidth / inputW;
      final scaleY = imageHeight / inputH;

      final x1 = (cx - w / 2) * scaleX;
      final y1 = (cy - h / 2) * scaleY;
      final x2 = (cx + w / 2) * scaleX;
      final y2 = (cy + h / 2) * scaleY;

      detections.add({
        'x1': x1.clamp(0.0, imageWidth.toDouble()),
        'y1': y1.clamp(0.0, imageHeight.toDouble()),
        'x2': x2.clamp(0.0, imageWidth.toDouble()),
        'y2': y2.clamp(0.0, imageHeight.toDouble()),
        'conf': maxScore,
        'classIdx': maxIdx,
        'name': maxIdx < classNames.length ? classNames[maxIdx] : 'Unknown',
      });
    }

    // Apply NMS
    return _nms(detections);
  }

  /// Non-maximum suppression
  List<Map<String, dynamic>> _nms(List<Map<String, dynamic>> detections) {
    if (detections.isEmpty) return detections;

    // Sort by confidence descending
    detections.sort(
      (a, b) => (b['conf'] as double).compareTo(a['conf'] as double),
    );

    final selected = <Map<String, dynamic>>[];
    final suppressed = List<bool>.filled(detections.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;
      selected.add(detections[i]);

      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(detections[i], detections[j]) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return selected;
  }

  /// Calculate Intersection over Union
  double _iou(Map<String, dynamic> a, Map<String, dynamic> b) {
    final x1 = max(a['x1'] as double, b['x1'] as double);
    final y1 = max(a['y1'] as double, b['y1'] as double);
    final x2 = min(a['x2'] as double, b['x2'] as double);
    final y2 = min(a['y2'] as double, b['y2'] as double);

    final intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1);
    final areaA = (a['x2'] - a['x1']) * (a['y2'] - a['y1']);
    final areaB = (b['x2'] - b['x1']) * (b['y2'] - b['y1']);
    final union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
  }

  /// Run MiDaS depth estimation
  Float32List _runMidasInference(
    Uint8List rgbBytes,
    int imageWidth,
    int imageHeight,
  ) {
    if (_midasInterpreter == null)
      return Float32List(midasInputSize * midasInputSize);

    // Preprocess: resize + normalize with MiDaS-specific values
    final input = Float32List(1 * midasInputSize * midasInputSize * 3);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (int y = 0; y < midasInputSize; y++) {
      for (int x = 0; x < midasInputSize; x++) {
        final srcX = (x * imageWidth / midasInputSize).toInt().clamp(
          0,
          imageWidth - 1,
        );
        final srcY = (y * imageHeight / midasInputSize).toInt().clamp(
          0,
          imageHeight - 1,
        );
        final srcIdx = (srcY * imageWidth + srcX) * 3;
        final dstIdx = (y * midasInputSize + x) * 3;

        input[dstIdx] = (rgbBytes[srcIdx] / 255.0 - mean[0]) / std[0];
        input[dstIdx + 1] = (rgbBytes[srcIdx + 1] / 255.0 - mean[1]) / std[1];
        input[dstIdx + 2] = (rgbBytes[srcIdx + 2] / 255.0 - mean[2]) / std[2];
      }
    }

    final inputTensor = input.reshape([1, midasInputSize, midasInputSize, 3]);
    final output = Float32List(
      midasInputSize * midasInputSize,
    ).reshape([1, midasInputSize, midasInputSize]);

    _midasInterpreter!.run(inputTensor, output);

    // Flatten output
    final depthMap = Float32List(midasInputSize * midasInputSize);
    for (int y = 0; y < midasInputSize; y++) {
      for (int x = 0; x < midasInputSize; x++) {
        depthMap[y * midasInputSize + x] = (output[0] as List)[y][x];
      }
    }

    return depthMap;
  }

  /// Clean up resources
  void dispose() {
    _yoloInterpreter?.close();
    _midasInterpreter?.close();
    _isInitialized = false;
  }
}
