import 'dart:ui';
import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Custom painter that draws detection bounding boxes on the camera preview
class DetectionOverlay extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;

  DetectionOverlay({required this.detections, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    for (final detection in detections) {
      // Scale bounding box to canvas size
      final rect = Rect.fromLTRB(
        detection.boundingBox.left * size.width,
        detection.boundingBox.top * size.height,
        detection.boundingBox.right * size.width,
        detection.boundingBox.bottom * size.height,
      );

      final color = _getColor(detection.distanceCategory);

      // Draw semi-transparent fill
      final fillPaint = Paint()
        ..color = color.withOpacity(0.15)
        ..style = PaintingStyle.fill;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        fillPaint,
      );

      // Draw border
      final borderPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        borderPaint,
      );

      // Draw label background
      final label = '${detection.className} · ${detection.distanceLabel}';
      final textStyle = TextStyle(
        color: Colors.white,
        fontSize: 13,
        fontWeight: FontWeight.w700,
        letterSpacing: 0.5,
        shadows: [
          Shadow(
            color: Colors.black54,
            blurRadius: 4,
            offset: const Offset(1, 1),
          ),
        ],
      );
      final textSpan = TextSpan(text: label, style: textStyle);
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final labelRect = Rect.fromLTWH(
        rect.left,
        rect.top - textPainter.height - 10,
        textPainter.width + 16,
        textPainter.height + 8,
      );

      // Label background with rounded corners
      final labelBgPaint = Paint()..color = color.withOpacity(0.85);
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(6)),
        labelBgPaint,
      );

      // Label text
      textPainter.paint(canvas, Offset(labelRect.left + 8, labelRect.top + 4));

      // Draw confidence badge
      final confLabel = '${(detection.confidence * 100).toInt()}%';
      final confStyle = TextStyle(
        color: Colors.white70,
        fontSize: 10,
        fontWeight: FontWeight.w500,
      );
      final confSpan = TextSpan(text: confLabel, style: confStyle);
      final confPainter = TextPainter(
        text: confSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      confPainter.paint(
        canvas,
        Offset(rect.right - confPainter.width - 6, rect.bottom + 4),
      );
    }
  }

  Color _getColor(DistanceCategory category) {
    switch (category) {
      case DistanceCategory.veryClose:
        return const Color(0xFFFF3B30); // Red
      case DistanceCategory.close:
        return const Color(0xFFFF9500); // Orange
      case DistanceCategory.nearby:
        return const Color(0xFFFFCC00); // Yellow
      case DistanceCategory.far:
        return const Color(0xFF34C759); // Green
    }
  }

  @override
  bool shouldRepaint(covariant DetectionOverlay oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
