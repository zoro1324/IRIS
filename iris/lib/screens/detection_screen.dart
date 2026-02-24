import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/detection_result.dart';
import '../services/tflite_service.dart';
import '../services/alert_service.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/alert_panel.dart';

/// Full-screen camera view with live obstacle detection
class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  final TfliteService _tfliteService = TfliteService();
  final AlertService _alertService = AlertService();

  List<DetectionResult> _detections = [];
  bool _isDetecting = false;
  bool _isCameraReady = false;
  bool _isProcessing = false;
  bool _midasEnabled = true;
  String _statusMessage = 'Initializing...';
  int _frameCount = 0;

  // Throttle: process every Nth frame (~5 FPS from 30 FPS stream)
  static const int _processEveryNthFrame = 6;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopDetection();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _stopDetection();
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initialize();
    }
  }

  Future<void> _initialize() async {
    // Request camera permission
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _statusMessage = 'Camera permission denied');
      if (mounted) {
        _alertService.speak(
          'Camera permission is required for obstacle detection.',
        );
      }
      return;
    }

    // Load TFLite models
    setState(() => _statusMessage = 'Loading AI models...');
    try {
      await _tfliteService.initialize();
      await _alertService.initialize();
      // Models loaded OK
    } catch (e) {
      setState(() => _statusMessage = 'Error loading models: $e');
      return;
    }

    // Initialize camera
    setState(() => _statusMessage = 'Starting camera...');
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _statusMessage = 'No camera found');
        return;
      }

      // Use back camera
      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        backCamera,
        ResolutionPreset.medium, // Balance quality vs performance
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();

      if (!mounted) return;
      setState(() {
        _isCameraReady = true;
        _statusMessage = 'Ready';
      });

      // Auto-start detection
      _startDetection();

      // Welcome message
      await _alertService.speak(
        'Obstacle detection started. Point your camera forward.',
      );
    } catch (e) {
      setState(() => _statusMessage = 'Camera error: $e');
    }
  }

  void _startDetection() {
    if (!_isCameraReady || _isDetecting) return;

    setState(() => _isDetecting = true);
    _frameCount = 0;

    _cameraController!.startImageStream((CameraImage image) {
      _frameCount++;
      if (_frameCount % _processEveryNthFrame != 0) return;
      if (_isProcessing) return;

      _processFrame(image);
    });
  }

  void _stopDetection() {
    if (_isDetecting) {
      _cameraController?.stopImageStream();
      setState(() {
        _isDetecting = false;
        _detections = [];
      });
      _alertService.stop();
      _alertService.resetCooldowns();
    }
  }

  Future<void> _processFrame(CameraImage image) async {
    _isProcessing = true;

    try {
      final results = await _tfliteService.detectObjects(
        image,
        image.width,
        image.height,
      );

      if (mounted) {
        setState(() => _detections = results);
        await _alertService.processDetections(results);
      }
    } catch (e) {
      print('Frame processing error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  void _goBack() {
    HapticFeedback.mediumImpact();
    _stopDetection();
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      body: _isCameraReady ? _buildCameraView() : _buildLoadingView(),
    );
  }

  Widget _buildLoadingView() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0A0E21), Color(0xFF1A1A2E)],
        ),
      ),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(
              width: 60,
              height: 60,
              child: CircularProgressIndicator(
                color: Color(0xFF6C63FF),
                strokeWidth: 3,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              _statusMessage,
              style: TextStyle(
                color: Colors.white.withOpacity(0.7),
                fontSize: 16,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 40),
            // Back button even during loading
            TextButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.arrow_back, color: Colors.white54),
              label: const Text(
                'Go Back',
                style: TextStyle(color: Colors.white54),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraView() {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera Preview
        ClipRect(
          child: OverflowBox(
            alignment: Alignment.center,
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: _cameraController!.value.previewSize!.height,
                height: _cameraController!.value.previewSize!.width,
                child: CameraPreview(_cameraController!),
              ),
            ),
          ),
        ),

        // Detection Overlay
        CustomPaint(
          painter: DetectionOverlay(
            detections: _detections,
            imageSize: Size(
              _cameraController!.value.previewSize!.height,
              _cameraController!.value.previewSize!.width,
            ),
          ),
          size: Size.infinite,
        ),

        // Top status bar
        Positioned(
          top: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: EdgeInsets.fromLTRB(
              16,
              MediaQuery.of(context).padding.top + 8,
              16,
              12,
            ),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.black.withOpacity(0.7),
                  Colors.black.withOpacity(0.0),
                ],
              ),
            ),
            child: Row(
              children: [
                // Back button
                Semantics(
                  label: 'Stop detection and go back',
                  button: true,
                  child: GestureDetector(
                    onTap: _goBack,
                    child: Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.arrow_back_ios_new_rounded,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                // Status indicator
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: _isDetecting
                        ? const Color(0xFF34C759).withOpacity(0.2)
                        : Colors.red.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: _isDetecting
                          ? const Color(0xFF34C759).withOpacity(0.5)
                          : Colors.red.withOpacity(0.5),
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _isDetecting
                              ? const Color(0xFF34C759)
                              : Colors.red,
                        ),
                      ),
                      const SizedBox(width: 6),
                      Text(
                        _isDetecting ? 'SCANNING' : 'PAUSED',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1,
                        ),
                      ),
                    ],
                  ),
                ),
                const Spacer(),
                // Detection count
                if (_detections.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      '${_detections.length} detected',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                const SizedBox(width: 8),
                // MiDaS toggle button
                Semantics(
                  label: _midasEnabled
                      ? 'Depth estimation on. Double tap to turn off.'
                      : 'Depth estimation off. Double tap to turn on.',
                  button: true,
                  child: GestureDetector(
                    onTap: () {
                      setState(() {
                        _midasEnabled = !_midasEnabled;
                        _tfliteService.midasEnabled = _midasEnabled;
                      });
                    },
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: _midasEnabled
                            ? const Color(0xFF6C63FF).withOpacity(0.3)
                            : Colors.white.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: _midasEnabled
                              ? const Color(0xFF6C63FF).withOpacity(0.6)
                              : Colors.white.withOpacity(0.15),
                        ),
                      ),
                      child: Icon(
                        Icons.layers_rounded,
                        color: _midasEnabled
                            ? const Color(0xFF6C63FF)
                            : Colors.white38,
                        size: 20,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),

        // Bottom alert panel
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AlertPanel(detections: _detections),
              // Stop button
              Container(
                color: Colors.black.withOpacity(0.85),
                padding: EdgeInsets.fromLTRB(
                  20,
                  12,
                  20,
                  MediaQuery.of(context).padding.bottom + 16,
                ),
                child: Semantics(
                  label: 'Stop obstacle detection. Double tap to stop.',
                  button: true,
                  child: SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      onPressed: _goBack,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(
                          0xFFFF3B30,
                        ).withOpacity(0.9),
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                        elevation: 0,
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.stop_rounded, size: 28),
                          SizedBox(width: 8),
                          Text(
                            'STOP DETECTION',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 2,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
