import 'dart:async';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:vibration/vibration.dart';
import '../models/detection_result.dart';

/// Manages TTS and haptic alerts for detected obstacles
class AlertService {
  static final AlertService _instance = AlertService._internal();
  factory AlertService() => _instance;
  AlertService._internal();

  final FlutterTts _tts = FlutterTts();
  bool _isInitialized = false;
  bool _isSpeaking = false;
  bool _vibrationEnabled = true;
  bool _ttsEnabled = true;

  /// Cooldown per class to prevent alert spam
  final Map<String, DateTime> _cooldowns = {};
  static const Duration _cooldownDuration = Duration(seconds: 3);
  static const Duration _urgentCooldownDuration = Duration(milliseconds: 1500);

  bool get isInitialized => _isInitialized;
  bool get vibrationEnabled => _vibrationEnabled;
  bool get ttsEnabled => _ttsEnabled;

  /// Initialize TTS engine
  Future<void> initialize() async {
    if (_isInitialized) return;

    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.55); // Slightly slow for clarity
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _tts.setCompletionHandler(() {
      _isSpeaking = false;
    });

    _isInitialized = true;
    print('Alert service initialized');
  }

  /// Toggle vibration feedback
  void setVibrationEnabled(bool enabled) {
    _vibrationEnabled = enabled;
  }

  /// Toggle TTS feedback
  void setTtsEnabled(bool enabled) {
    _ttsEnabled = enabled;
  }

  /// Process detection results and trigger alerts
  Future<void> processDetections(List<DetectionResult> detections) async {
    if (detections.isEmpty) return;

    // Find the most urgent detection that isn't in cooldown
    DetectionResult? urgentDetection;

    for (final det in detections) {
      final key = '${det.className}_${det.distanceCategory.name}';
      final lastAlert = _cooldowns[key];
      final cooldown = det.distanceCategory == DistanceCategory.veryClose
          ? _urgentCooldownDuration
          : _cooldownDuration;

      if (lastAlert == null ||
          DateTime.now().difference(lastAlert) > cooldown) {
        urgentDetection = det;
        _cooldowns[key] = DateTime.now();
        break;
      }
    }

    if (urgentDetection == null) return;

    // Trigger haptic feedback based on proximity
    if (_vibrationEnabled) {
      _triggerHaptic(urgentDetection.distanceCategory);
    }

    // Speak alert
    if (_ttsEnabled && !_isSpeaking) {
      await _speakAlert(urgentDetection);
    }
  }

  /// Trigger haptic feedback based on distance
  void _triggerHaptic(DistanceCategory category) async {
    final hasVibrator = await Vibration.hasVibrator();
    if (hasVibrator != true) return;

    switch (category) {
      case DistanceCategory.veryClose:
        // Strong continuous vibration
        Vibration.vibrate(duration: 500, amplitude: 255);
        break;
      case DistanceCategory.close:
        // Medium pulse
        Vibration.vibrate(duration: 300, amplitude: 180);
        break;
      case DistanceCategory.nearby:
        // Light pulse
        Vibration.vibrate(duration: 150, amplitude: 100);
        break;
      case DistanceCategory.far:
        // No vibration for far objects
        break;
    }
  }

  /// Speak alert about detected obstacle
  Future<void> _speakAlert(DetectionResult detection) async {
    _isSpeaking = true;

    String message;
    switch (detection.distanceCategory) {
      case DistanceCategory.veryClose:
        message = 'Warning! ${detection.className} very close!';
        break;
      case DistanceCategory.close:
        message = '${detection.className} close ahead';
        break;
      case DistanceCategory.nearby:
        message = '${detection.className} nearby';
        break;
      case DistanceCategory.far:
        message = '${detection.className} ahead';
        break;
    }

    await _tts.speak(message);
  }

  /// Speak a custom message (for onboarding, etc.)
  Future<void> speak(String message) async {
    await _tts.speak(message);
  }

  /// Stop ongoing speech
  Future<void> stop() async {
    await _tts.stop();
    _isSpeaking = false;
  }

  /// Reset cooldowns
  void resetCooldowns() {
    _cooldowns.clear();
  }

  /// Clean up resources
  void dispose() {
    _tts.stop();
    _cooldowns.clear();
    _isInitialized = false;
  }
}
