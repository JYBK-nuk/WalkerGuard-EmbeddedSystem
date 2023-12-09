import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';

void main() => runApp(MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  void initState() {
    super.initState();
    _initStateAsync();
  }

  /// List of available cameras
  late List<CameraDescription> cameras;

  /// Controller
  CameraController? _cameraController;

  get _controller => _cameraController;
  void _initStateAsync() async {
    _initializeCamera();
  }

  void _initializeCamera() async {
    cameras = await availableCameras();
    // cameras[0] for back-camera
    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
    )..initialize().then((_) async {
        setState(() {});
      });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        useMaterial3: true,
      ),
      home: Scaffold(
          appBar: AppBar(
            title: const Text('撞到人要說對不起'),
            centerTitle: true,
          ),
          body: (_cameraController == null || !_controller.value.isInitialized)
              ? const Center(child: CircularProgressIndicator())
              : CameraPreview_(controller: _controller)),
    );
  }
}

class CameraPreview_ extends StatefulWidget {
  final controller;

  const CameraPreview_({
    Key? key,
    required this.controller,
  }) : super(key: key);

  @override
  _CameraPreviewState createState() => _CameraPreviewState();
}

class _CameraPreviewState extends State<CameraPreview_> {
  @override
  Widget build(BuildContext context) {
    if (widget.controller == null || !widget.controller.value.isInitialized)
      return SizedBox.shrink();
    var aspect = 1 / widget.controller.value.aspectRatio;
    return Stack(
      children: [
        AspectRatio(
          aspectRatio: aspect,
          child: CameraPreview(widget.controller),
        ),
      ],
    );
  }
}
