/**
 * MistralApp.swift
 * SwiftUI Application for Mistral Small 3.2 Inference
 */

import SwiftUI
import AppKit
import MistralCore

@main
struct MistralApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var modelManager = ModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
                .frame(minWidth: 800, minHeight: 600)
        }
        .defaultSize(width: 1000, height: 700)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(replacing: .appInfo) {
                Button("About Mistral") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            .applicationName: "Mistral",
                            .applicationVersion: "1.0",
                            .credits: NSAttributedString(string: "Mistral Small 3.2 inference powered by MLX Swift")
                        ]
                    )
                }
            }
        }

        Settings {
            SettingsView()
                .environmentObject(modelManager)
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Make this a regular app that appears in the Dock
        NSApplication.shared.setActivationPolicy(.regular)

        // Bring to front
        NSApplication.shared.activate(ignoringOtherApps: true)

        // Make the first window key and front
        if let window = NSApplication.shared.windows.first {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
