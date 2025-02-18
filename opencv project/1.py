# import cv2
# import numpy as np


# def detect_skin_tone(frame, face_cascade):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
#     detected_skin_tone = None
    
#     for (x, y, w, h) in faces:
#         cheek_width = int(w * 0.2)
#         cheek_height = int(h * 0.2)
        
#         left_cheek_x1 = x + int(w * 0.1)
#         left_cheek_y1 = y + int(h * 0.5)
#         right_cheek_x1 = x + int(w * 0.7)
#         right_cheek_y1 = y + int(h * 0.5)
        
#         left_cheek = frame[left_cheek_y1:left_cheek_y1 + cheek_height, left_cheek_x1:left_cheek_x1 + cheek_width]
#         right_cheek = frame[right_cheek_y1:right_cheek_y1 + cheek_height, right_cheek_x1:right_cheek_x1 + cheek_width]
        
#         if left_cheek.size > 0 and right_cheek.size > 0:
#             hsv_left = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2HSV)
#             hsv_right = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2HSV)
            
#             v_mean_left = np.mean(hsv_left[:, :, 2])
#             v_mean_right = np.mean(hsv_right[:, :, 2])
#             v_mean = (v_mean_left + v_mean_right) / 2
            
#             if v_mean > 180:
#                 detected_skin_tone = "Fair"
#                 color = (255, 255, 255)
#             elif v_mean > 120:
#                 detected_skin_tone = "Medium"
#                 color = (0, 255, 255)
#             else:
#                 detected_skin_tone = "Deep"
#                 color = (0, 0, 255)
            
#             cv2.rectangle(frame, (left_cheek_x1, left_cheek_y1),
#                           (left_cheek_x1 + cheek_width, left_cheek_y1 + cheek_height), color, 2)
#             cv2.rectangle(frame, (right_cheek_x1, right_cheek_y1),
#                           (right_cheek_x1 + cheek_width, right_cheek_y1 + cheek_height), color, 2)
            
#             cv2.putText(frame, f"Skin Tone: {detected_skin_tone}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     return frame, detected_skin_tone  # Return both frame and detected tone

# def take_photo(face_cascade):
#     cap = cv2.VideoCapture(0)  # Use the default camera
    
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return
    
#     print("Press 'Enter' to capture photo.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Display the frame live
#         cv2.imshow("Press 'Enter' to capture photo", frame)
        
#         # Wait for the user to press Enter
#         if cv2.waitKey(1) & 0xFF == ord('\r'):
#             print("Photo captured.")
#             processed_frame, skin_tone = detect_skin_tone(frame, face_cascade)
#             if skin_tone:
#                 print(f"Detected Skin Tone: {skin_tone}")
#             cv2.imshow("Captured Photo - Skin Tone Detection", processed_frame)
#             cv2.waitKey(0)  # Wait until any key is pressed to close the image window
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
#     # Ask user to choose option
#     choice = input("Choose option (1 for Capture and Detect Skin Tone): ")
    
#     if choice == "1":
#         take_photo(face_cascade)
#     else:
#         print("Invalid choice. Exiting.")

import os
import cv2

# Path to the haarcascades directory
haar_cascades_path = cv2.data.haarcascades

# List all files in the haarcascades directory
files = os.listdir(haar_cascades_path)

# Check if the specific classifier is in the list
classifier_name = "haarcascade_frontalface_default.xml"
if classifier_name in files:
    print(f"{classifier_name} is available.")
else:
    print(f"{classifier_name} is not available.")
    
# You can also print all available files to inspect
print("\nList of available classifiers:")
print(files)
