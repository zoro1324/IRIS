import 'dart:async';
import 'dart:typed_data';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/detection_result.dart';
import 'dart:ui';

/// Plane data copied out of a CameraImage before any async work.
class CameraPlane {
  final Uint8List bytes;
  final int bytesPerRow;
  const CameraPlane({required this.bytes, required this.bytesPerRow});
}

/// Service for running TFLite inference (YOLO + MiDaS).
///
/// Architecture (Mali-GPU-safe):
///  • Plain [Interpreter] on the main isolate — no IsolateInterpreter
///    (which crashes Mali GPUs by sharing native memory across isolates).
///  • The detection_screen STOPS the camera image stream before calling
///    detectObjects(), and RESTARTS it after. This ensures:
///      1. No camera buffer contention during inference
///      2. The GPU isn't rendering camera frames while TFLite runs
///  • Frame rate is throttled via a Timer rather than frame counting.
class TfliteService {
  static final TfliteService _instance = TfliteService._internal();
  factory TfliteService() => _instance;
  TfliteService._internal();

  Interpreter? _yoloInterpreter;
  Interpreter? _midasInterpreter;
  bool _isInitialized = false;
  bool midasEnabled = true;

  /// COCO 80-class names
  static const List<String> classNames = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
  ];

  static const int midasInputSize = 256;
  static const double _confidenceThreshold = 0.4;
  static const double _iouThreshold = 0.45;

  bool get isInitialized => _isInitialized;

  // ─── Initialization ──────────────────────────────────────────────────────────

  Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      _yoloInterpreter = await Interpreter.fromAsset(
        'assets/yolov8n.tflite',
        options: InterpreterOptions()..threads = 4,
      );
      print('YOLO loaded');
      print('  input : ${_yoloInterpreter!.getInputTensor(0).shape}');
      print('  output: ${_yoloInterpreter!.getOutputTensor(0).shape}');

      _midasInterpreter = await Interpreter.fromAsset(
        'assets/midas_v21_small_256.tflite',
        options: InterpreterOptions()..threads = 2,
      );
      print('MiDaS loaded');
      _isInitialized = true;
    } catch (e) {
      print('Error loading models: $e');
      rethrow;
    }
  }

  // ─── Public API ──────────────────────────────────────────────────────────────

  /// Run detection on pre-copied plane data.
  ///
  /// IMPORTANT: The caller (detection_screen) must STOP the camera image
  /// stream BEFORE calling this, and RESTART it AFTER this returns. This
  /// prevents Mali GPU contention between the camera HAL and TFLite.
  Future<List<DetectionResult>> detectObjects(
    int imageWidth,
    int imageHeight,
    ImageFormatGroup format,
    List<CameraPlane> planes,
  ) async {
    if (!_isInitialized) return [];

    try {
      // 1. YUV→RGB
      final rgbBytes = _convertToRgb(imageWidth, imageHeight, format, planes);
      if (rgbBytes == null) return [];

      // 2. YOLO inference (synchronous, but camera stream is stopped)
      final detections = _runYolo(rgbBytes, imageWidth, imageHeight);

      // 3. MiDaS inference (optional)
      Float32List? depthMap;
      if (midasEnabled) {
        depthMap = _runMidas(rgbBytes, imageWidth, imageHeight);
      }

      // 4. Combine
      final results = <DetectionResult>[];
      for (final det in detections) {
        double distance = 999.0;
        if (depthMap != null) {
          final cx = ((det['x1'] + det['x2']) / 2)
              .clamp(0, imageWidth - 1)
              .toInt();
          final cy = ((det['y1'] + det['y2']) / 2)
              .clamp(0, imageHeight - 1)
              .toInt();
          final dx = (cx * midasInputSize / imageWidth).toInt().clamp(
            0,
            midasInputSize - 1,
          );
          final dy = (cy * midasInputSize / imageHeight).toInt().clamp(
            0,
            midasInputSize - 1,
          );
          final dv = depthMap[dy * midasInputSize + dx];
          distance = dv > 0 ? 1000.0 / dv : 999.0;
        }
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
      results.sort((a, b) => a.priority.compareTo(b.priority));
      return results;
    } catch (e) {
      print('Detection error: $e');
      return [];
    }
  }

  // ─── YOLO inference ──────────────────────────────────────────────────────────

  List<Map<String, dynamic>> _runYolo(
    Uint8List rgbBytes,
    int imageWidth,
    int imageHeight,
  ) {
    if (_yoloInterpreter == null) return [];
    final inputShape = _yoloInterpreter!.getInputTensor(0).shape;
    final inputH = inputShape[1];
    final inputW = inputShape[2];

    final input = Float32List(1 * inputH * inputW * 3);
    for (int y = 0; y < inputH; y++) {
      for (int x = 0; x < inputW; x++) {
        final sx = (x * imageWidth / inputW).toInt().clamp(0, imageWidth - 1);
        final sy = (y * imageHeight / inputH).toInt().clamp(0, imageHeight - 1);
        final si = (sy * imageWidth + sx) * 3;
        final di = (y * inputW + x) * 3;
        input[di] = rgbBytes[si] / 255.0;
        input[di + 1] = rgbBytes[si + 1] / 255.0;
        input[di + 2] = rgbBytes[si + 2] / 255.0;
      }
    }

    final inputTensor = input.reshape([1, inputH, inputW, 3]);
    final outputShape = _yoloInterpreter!.getOutputTensor(0).shape;
    final dim1 = outputShape[1];
    final dim2 = outputShape[2];

    final output = List.generate(
      1,
      (_) => List.generate(dim1, (_) => List<double>.filled(dim2, 0.0)),
    );
    _yoloInterpreter!.run(inputTensor, output);

    final flat = Float32List(dim1 * dim2);
    int idx = 0;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        flat[idx++] = output[0][i][j];
      }
    }
    return _parseYolo(
      flat,
      outputShape,
      imageWidth,
      imageHeight,
      inputW,
      inputH,
    );
  }

  // ─── Output parsing ──────────────────────────────────────────────────────────

  List<Map<String, dynamic>> _parseYolo(
    Float32List flat,
    List<int> shape,
    int imageWidth,
    int imageHeight,
    int inputW,
    int inputH,
  ) {
    if (shape.length != 3) return [];
    final nc = classNames.length;
    final dim1 = shape[1], dim2 = shape[2];

    bool transposed;
    int numDet;
    if (dim2 == 4 + nc) {
      transposed = false;
      numDet = dim1;
    } else if (dim1 == 4 + nc) {
      transposed = true;
      numDet = dim2;
    } else {
      transposed = dim2 > dim1;
      numDet = transposed ? dim2 : dim1;
    }

    final detections = <Map<String, dynamic>>[];
    for (int i = 0; i < numDet; i++) {
      double cx, cy, bw, bh, maxScore = -1;
      int maxIdx = 0;
      if (transposed) {
        cx = flat[0 * numDet + i];
        cy = flat[1 * numDet + i];
        bw = flat[2 * numDet + i];
        bh = flat[3 * numDet + i];
        for (int c = 0; c < nc; c++) {
          final s = flat[(4 + c) * numDet + i];
          if (s > maxScore) {
            maxScore = s;
            maxIdx = c;
          }
        }
      } else {
        final stride = 4 + nc;
        cx = flat[i * stride];
        cy = flat[i * stride + 1];
        bw = flat[i * stride + 2];
        bh = flat[i * stride + 3];
        for (int c = 0; c < nc; c++) {
          final s = flat[i * stride + 4 + c];
          if (s > maxScore) {
            maxScore = s;
            maxIdx = c;
          }
        }
      }
      if (maxScore < _confidenceThreshold) continue;
      final scaleX = imageWidth / inputW;
      final scaleY = imageHeight / inputH;
      detections.add({
        'x1': ((cx - bw / 2) * scaleX).clamp(0.0, imageWidth.toDouble()),
        'y1': ((cy - bh / 2) * scaleY).clamp(0.0, imageHeight.toDouble()),
        'x2': ((cx + bw / 2) * scaleX).clamp(0.0, imageWidth.toDouble()),
        'y2': ((cy + bh / 2) * scaleY).clamp(0.0, imageHeight.toDouble()),
        'conf': maxScore,
        'classIdx': maxIdx,
        'name': maxIdx < nc ? classNames[maxIdx] : 'Unknown',
      });
    }
    return _nms(detections);
  }

  // ─── NMS ─────────────────────────────────────────────────────────────────────

  List<Map<String, dynamic>> _nms(List<Map<String, dynamic>> dets) {
    if (dets.isEmpty) return dets;
    dets.sort((a, b) => (b['conf'] as double).compareTo(a['conf'] as double));
    final selected = <Map<String, dynamic>>[];
    final suppressed = List<bool>.filled(dets.length, false);
    for (int i = 0; i < dets.length; i++) {
      if (suppressed[i]) continue;
      selected.add(dets[i]);
      for (int j = i + 1; j < dets.length; j++) {
        if (!suppressed[j]) {
          final ax1 = dets[i]['x1'] as double, ay1 = dets[i]['y1'] as double;
          final ax2 = dets[i]['x2'] as double, ay2 = dets[i]['y2'] as double;
          final bx1 = dets[j]['x1'] as double, by1 = dets[j]['y1'] as double;
          final bx2 = dets[j]['x2'] as double, by2 = dets[j]['y2'] as double;
          final inter =
              max(0.0, min(ax2, bx2) - max(ax1, bx1)) *
              max(0.0, min(ay2, by2) - max(ay1, by1));
          final union =
              (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
          if (union > 0 && inter / union > _iouThreshold) suppressed[j] = true;
        }
      }
    }
    return selected;
  }

  // ─── YUV → RGB ───────────────────────────────────────────────────────────────

  Uint8List? _convertToRgb(
    int w,
    int h,
    ImageFormatGroup format,
    List<CameraPlane> planes,
  ) {
    try {
      final rgb = Uint8List(w * h * 3);
      if (format == ImageFormatGroup.yuv420 && planes.length >= 2) {
        final yP = planes[0], uP = planes[1];
        final hasV = planes.length >= 3;
        final vP = hasV ? planes[2] : planes[1];
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final yi = y * yP.bytesPerRow + x;
            final uvi = (y ~/ 2) * uP.bytesPerRow + (x ~/ 2);
            final yv = yP.bytes[yi];
            final uv = uP.bytes[uvi];
            final vv = hasV ? vP.bytes[uvi] : uP.bytes[uvi + 1];
            final idx = (y * w + x) * 3;
            rgb[idx] = (yv + 1.370705 * (vv - 128)).clamp(0, 255).toInt();
            rgb[idx + 1] = (yv - 0.337633 * (uv - 128) - 0.698001 * (vv - 128))
                .clamp(0, 255)
                .toInt();
            rgb[idx + 2] = (yv + 1.732446 * (uv - 128)).clamp(0, 255).toInt();
          }
        }
      } else if (format == ImageFormatGroup.bgra8888 && planes.isNotEmpty) {
        final bytes = planes[0].bytes;
        for (int i = 0, j = 0; i < bytes.length; i += 4, j += 3) {
          rgb[j] = bytes[i + 2];
          rgb[j + 1] = bytes[i + 1];
          rgb[j + 2] = bytes[i];
        }
      }
      return rgb;
    } catch (e) {
      print('Image conversion error: $e');
      return null;
    }
  }

  // ─── Cleanup ─────────────────────────────────────────────────────────────────

  void dispose() {
    _yoloInterpreter?.close();
    _midasInterpreter?.close();
    _isInitialized = false;
  }

  // ─── MiDaS inference ─────────────────────────────────────────────────────────

  Float32List _runMidas(Uint8List rgbBytes, int imageWidth, int imageHeight) {
    const sz = midasInputSize;
    if (_midasInterpreter == null) return Float32List(sz * sz);
    final input = Float32List(1 * sz * sz * 3);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    for (int y = 0; y < sz; y++) {
      for (int x = 0; x < sz; x++) {
        final sx = (x * imageWidth / sz).toInt().clamp(0, imageWidth - 1);
        final sy = (y * imageHeight / sz).toInt().clamp(0, imageHeight - 1);
        final si = (sy * imageWidth + sx) * 3;
        final di = (y * sz + x) * 3;
        input[di] = (rgbBytes[si] / 255.0 - mean[0]) / std[0];
        input[di + 1] = (rgbBytes[si + 1] / 255.0 - mean[1]) / std[1];
        input[di + 2] = (rgbBytes[si + 2] / 255.0 - mean[2]) / std[2];
      }
    }
    final inputTensor = input.reshape([1, sz, sz, 3]);
    final output = List.generate(
      1,
      (_) => List.generate(sz, (_) => List<double>.filled(sz, 0.0)),
    );
    _midasInterpreter!.run(inputTensor, output);
    final depth = Float32List(sz * sz);
    for (int y = 0; y < sz; y++) {
      for (int x = 0; x < sz; x++) {
        depth[y * sz + x] = output[0][y][x];
      }
    }
    return depth;
  }
}
