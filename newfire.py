from ultralytics import YOLO
import cvzone
import cv2
import math
import os

# Load the YOLO fire detection model
model = YOLO('fire.pt')

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Change to 'fire2.mp4' for video file input

# Reading class names
classnames = ['fire']

# Create folder to store detected fire images
output_folder = "detected_fire_images"
os.makedirs(output_folder, exist_ok=True)
frame_count = 0  # Counter to save images with unique names

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access camera or video.")
        break

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Process detected objects   
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            if confidence > 50:  # Detect fire with high confidence
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

                # Save the detected fire frame as an image
                frame_count += 1
                image_filename = os.path.join(output_folder, f"fire_detected_{frame_count}.jpg")
                cv2.imwrite(image_filename, frame)
                print(f"Saved detected fire image: {image_filename}")

    # Show the live detection results
    cv2.imshow('Fire Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
