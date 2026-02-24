import 'dart:ui';

/// Distance category based on depth estimation
enum DistanceCategory {
  veryClose, // Immediate danger
  close, // Warning
  nearby, // Caution
  far, // Safe
}

/// Result from a single object detection
class DetectionResult {
  final String className;
  final double confidence;
  final Rect boundingBox;
  final double depthValue;
  final DistanceCategory distanceCategory;

  DetectionResult({
    required this.className,
    required this.confidence,
    required this.boundingBox,
    required this.depthValue,
    required this.distanceCategory,
  });

  /// Get human-readable distance label
  String get distanceLabel {
    switch (distanceCategory) {
      case DistanceCategory.veryClose:
        return 'Very Close';
      case DistanceCategory.close:
        return 'Close';
      case DistanceCategory.nearby:
        return 'Nearby';
      case DistanceCategory.far:
        return 'Far';
    }
  }

  /// Get alert priority (lower = more urgent)
  int get priority {
    switch (distanceCategory) {
      case DistanceCategory.veryClose:
        return 0;
      case DistanceCategory.close:
        return 1;
      case DistanceCategory.nearby:
        return 2;
      case DistanceCategory.far:
        return 3;
    }
  }

  /// Categorize depth value into distance category
  /// MiDaS gives inverse depth: higher value = closer object
  static DistanceCategory categorizeDepth(double distance) {
    if (distance < 3.0) return DistanceCategory.veryClose;
    if (distance < 8.0) return DistanceCategory.close;
    if (distance < 15.0) return DistanceCategory.nearby;
    return DistanceCategory.far;
  }
}
