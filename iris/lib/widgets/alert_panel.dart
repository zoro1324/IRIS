import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Bottom panel showing currently detected obstacles sorted by distance
class AlertPanel extends StatelessWidget {
  final List<DetectionResult> detections;

  const AlertPanel({super.key, required this.detections});

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.black.withOpacity(0.0),
              Colors.black.withOpacity(0.7),
            ],
          ),
        ),
        child: const Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.check_circle_outline, color: Colors.green, size: 20),
            SizedBox(width: 8),
            Text(
              'Path is clear',
              style: TextStyle(
                color: Colors.white70,
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      );
    }

    return Container(
      constraints: const BoxConstraints(maxHeight: 160),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.black.withOpacity(0.0),
            Colors.black.withOpacity(0.85),
          ],
        ),
      ),
      child: ListView.builder(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        itemCount: detections.length > 3 ? 3 : detections.length,
        itemBuilder: (context, index) {
          final det = detections[index];
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: _AlertItem(detection: det),
          );
        },
      ),
    );
  }
}

class _AlertItem extends StatelessWidget {
  final DetectionResult detection;

  const _AlertItem({required this.detection});

  @override
  Widget build(BuildContext context) {
    final color = _getColor(detection.distanceCategory);
    final icon = _getIcon(detection.className);

    return Semantics(
      label: '${detection.className} ${detection.distanceLabel}',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.4), width: 1),
        ),
        child: Row(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    detection.className,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    detection.distanceLabel,
                    style: TextStyle(
                      color: color,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '${(detection.confidence * 100).toInt()}%',
                style: TextStyle(
                  color: color,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  IconData _getIcon(String className) {
    switch (className.toLowerCase()) {
      case 'car':
        return Icons.directions_car;
      case 'bus':
        return Icons.directions_bus;
      case 'truck':
        return Icons.local_shipping;
      case 'motorcycle':
        return Icons.two_wheeler;
      case 'bicycle':
        return Icons.pedal_bike;
      case 'person':
        return Icons.person;
      case 'chair':
        return Icons.chair;
      case 'table':
        return Icons.table_restaurant;
      case 'stair':
        return Icons.stairs;
      case 'cow':
        return Icons.pets;
      case 'dogs':
        return Icons.pets;
      case 'trash':
        return Icons.delete;
      default:
        return Icons.warning;
    }
  }

  Color _getColor(DistanceCategory category) {
    switch (category) {
      case DistanceCategory.veryClose:
        return const Color(0xFFFF3B30);
      case DistanceCategory.close:
        return const Color(0xFFFF9500);
      case DistanceCategory.nearby:
        return const Color(0xFFFFCC00);
      case DistanceCategory.far:
        return const Color(0xFF34C759);
    }
  }
}
