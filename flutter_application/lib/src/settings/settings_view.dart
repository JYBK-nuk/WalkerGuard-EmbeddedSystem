import 'package:flutter/material.dart';
import 'settings_controller.dart';
class SettingsView extends StatelessWidget {
  const SettingsView({super.key, required this.controller});

  static const routeName = '/settings';

  final SettingsController controller;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('設定'),
      ),
      body: ListView(
        children: [
          ListTile(
            title: const Text('主題'),
            subtitle: const Text('應用程式主題'),
            trailing: DropdownButton<ThemeMode>(
              value: controller.themeMode,
              onChanged: controller.updateThemeMode,
              items: const [
                DropdownMenuItem(
                  value: ThemeMode.system,
                  child: Text('跟隨系統'),
                ),
                DropdownMenuItem(
                  value: ThemeMode.light,
                  child: Text('亮色主題'),
                ),
                DropdownMenuItem(
                  value: ThemeMode.dark,
                  child: Text('暗色主題'),
                )
              ],
            ),
          ),

          ListTile(
            title: const Text('API 連結'),
            subtitle: const Text('API url 連結'),
            trailing: TextField(
              onChanged: controller.updateApiUrl,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'API 連結',
              ),
            ),
          ),
        ],
      ),
    );
  }
}
