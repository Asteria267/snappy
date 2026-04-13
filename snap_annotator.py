import cv2
import ollama
import base64
from PIL import Image
import io
import json
import time

print("🚀 SnapAnnotator - Day 12")
print("Press 's' to capture and analyze")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam. Close other camera apps and try again.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 's' to snapshot", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("SnapAnnotator - Live Feed", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        print("\n📸 Capturing and analyzing with moondream...")
        start_time = time.time()

        # Resize for speed
        resized = cv2.resize(frame, (512, 512))
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Smart prompt - asks for clean, usable output
        response = ollama.chat(
            model="moondream",
            messages=[{
                'role': 'user',
                'content': '''Analyze this image carefully. 
Return ONLY valid JSON in this exact format:
{
  "description": "One clear sentence describing the whole scene",
  "objects": [
    {"name": "object name", "position": "left/center/right/top/bottom", "details": "short useful description"}
  ]
}
Be accurate and concise.''',
                'images': [img_base64]
            }]
        )

        try:
            result = json.loads(response['message']['content'])
            
            print("\n" + "="*50)
            print("📝 ANALYSIS")
            print("="*50)
            print("Description:", result.get("description", "No description"))
            print("\nDetected Objects:")
            
            annotated = frame.copy()
            
            for i, obj in enumerate(result.get("objects", [])):
                name = obj.get("name", "unknown")
                pos = obj.get("position", "center")
                details = obj.get("details", "")
                
                print(f"  {i+1}. {name} ({pos}) - {details}")
                
                # Simple visual: draw a box and label (smart but not exaggerated)
                h, w = frame.shape[:2]
                x = w // 2 if "center" in pos else (w//4 if "left" in pos else 3*w//4)
                y = h // 2 if "center" in pos else (h//4 if "top" in pos else 3*h//4)
                
                cv2.rectangle(annotated, (x-60, y-40), (x+60, y+40), (0, 255, 0), 2)
                cv2.putText(annotated, name, (x-50, y-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Annotated Snapshot", annotated)
            
            latency = time.time() - start_time
            print(f"\n⏱️  Done in {latency:.1f} seconds")

            # Smart follow-up
            print("\n💬 Ask a follow-up question about the image (or press Enter to skip):")
            question = input("> ").strip()
            if question:
                print("Thinking...")
                follow_response = ollama.chat(
                    model="moondream",
                    messages=[{
                        'role': 'user',
                        'content': f"{question}\nImage description: {result.get('description')}",
                        'images': [img_base64]
                    }]
                )
                print("\nAnswer:", follow_response['message']['content'])

        except Exception as e:
            print("❌ Could not parse JSON. Raw response below:")
            print(response['message']['content'])

print("\n👋 SnapAnnotator closed.")
cap.release()
cv2.destroyAllWindows()
