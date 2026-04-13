📸 SnapAnnotator

SnapAnnotator is a lightweight real-time computer vision tool that captures webcam images and uses an AI vision model to generate structured, human-readable annotations.

It provides both a high-level scene description and a list of detected objects, along with simple visual overlays—making it useful for learning, prototyping, and interactive demos.

🚀 Features
📷 Real-time webcam capture
🧠 AI-powered image understanding (via Moondream)
📝 Structured JSON output (description + objects)
🟩 Visual annotations on detected objects
💬 Interactive follow-up Q&A on captured images
⚡ Optimized for speed (resized inference)
🛠️ Tech Stack
Python
OpenCV
PIL (Pillow)
Ollama (Moondream model)
▶️ How It Works
Launch the app
Press s to capture a snapshot
The image is analyzed by the AI model
You receive:
A one-line scene description
A list of detected objects with positions
Annotated image is displayed
Optionally ask follow-up questions about the image
⌨️ Controls
s → Capture & analyze image
q → Quit application
📦 Installation
pip install opencv-python pillow ollama

Make sure you have Ollama running with the moondream model available.

▶️ Run
python snap_annotator.py
📌 Notes
Designed to be lightweight and fast
Uses simplified positioning instead of full object detection boxes
Requires a working webcam
💡 Use Cases
Learning computer vision basics
AI demos & hackathons
Rapid prototyping
Interactive image understanding
