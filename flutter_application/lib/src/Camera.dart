import 'package:flutter/material.dart';
import 'settings/settings_view.dart';
import 'package:camera/camera.dart';

import 'dart:io';

/// Displays a list of SampleItems.
class CameraView extends StatefulWidget {
  const CameraView({
    super.key,
    required this.cameras,
  });

  static const routeName = '/';
  final List<CameraDescription> cameras;

  @override
  State<CameraView> createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize().then((value) async {
      // FlashMode currentFlashMode = _controller.value.flashMode;
      // if (currentFlashMode != null)
      try {
        await _controller.setFlashMode(FlashMode.off);
      } catch (error) {}
    });
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CameraView'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.restorablePushNamed(context, SettingsView.routeName);
            },
          ),
        ],
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // If the Future is complete, display the preview.
            return CameraPreview(_controller);
          } else {
            // Otherwise, display a loading indicator.
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        // Provide an onPressed callback.
        onPressed: takePicture,
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  Future takePicture() async {
    if (!_controller.value.isInitialized) {
      return null;
    }
    if (_controller.value.isTakingPicture) {
      debugPrint('isTakingPicture');
      return null;
    }
    try {
      debugPrint('takePicture');
      await _controller.setFlashMode(FlashMode.off);
      XFile picture = await _controller.takePicture();
      Navigator.push(
          context,
          MaterialPageRoute(
              builder: (context) => PreviewPage(
                    picture: picture,
                  )));
    } on CameraException catch (e) {
      debugPrint('Error occured while taking picture: $e');
      return null;
    }
  }
}

class PreviewPage extends StatelessWidget {
  const PreviewPage({super.key, required this.picture});

  final XFile picture;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PreviewPage'),
      ),
      body: Center(
        child: Image.file(File(picture.path)),
      ),
    );
  }
}
