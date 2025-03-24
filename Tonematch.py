import cv2
import numpy as np

def detect_skin_and_tone(image):
    """
    Detects skin regions in the given image and classifies skin tone.
    """
    # Convert the image to YCrCb color space
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define the skin color range in YCrCb
    lower_bound = np.array([0, 130, 75], dtype=np.uint8)
    upper_bound = np.array([255, 180, 135], dtype=np.uint8)
    
    # Create a binary mask for skin color
    skin_mask = cv2.inRange(ycrcb_image, lower_bound, upper_bound)
    
    # Apply the mask to extract skin regions
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    
    # Analyze the Y (luminance) channel to classify skin tone
    y_channel = ycrcb_image[:, :, 0]
    masked_y = cv2.bitwise_and(y_channel, y_channel, mask=skin_mask)
    non_zero_y = masked_y[masked_y > 0]
    
    if len(non_zero_y) > 0:
        average_brightness = np.mean(non_zero_y)
        if average_brightness > 135:
            skin_tone = "Fair"
        elif 90 <= average_brightness <= 135:
            skin_tone = "Medium"
        else:
            skin_tone = "Deep"
    else:
        skin_tone = "No skin detected"
    
    return skin, skin_mask, average_brightness, skin_tone

def recommend_shirt_colors(skin_tone):
    """
    Suggests the best shirt colors based on skin tone.
    """
    recommendations = {
        "Deep": ["White", "Light Blue", "Olive Green", "Burgundy", "Mustard Yellow", "Deep Purple"],
        "Medium": ["Teal", "Rich Green", "Orange", "Warm Grey", "Royal Blue", "Maroon"],
        "Fair": ["Navy Blue", "Deep Green", "Charcoal Grey", "Burgundy", "Earthy Tones"]
    }
    return recommendations.get(skin_tone, ["No recommendation available"])

def main():
    print("Press 'q' to capture a frame and detect skin tone.")
    cap = cv2.VideoCapture(0)  # Capture video from the webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access the webcam.")
            break
        
        cv2.imshow("Live Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            skin, _, avg_brightness, skin_tone = detect_skin_and_tone(frame)
            print(f"Skin Colour: {skin_tone}")
            
            shirt_colors = recommend_shirt_colors(skin_tone)
            print("Recommended Shirt Colors:")
            for color in shirt_colors:
                print(f"✅ {color}")
            
            y_offset = 60
            for color in shirt_colors:
                cv2.putText(frame, f"✅ {color}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 40  # Spacing between each recommendation
            
            cv2.putText(frame, f"Skin Tone: {skin_tone}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Captured Frame - Skin Detection", frame)
            break  # Exit the loop after capturing one frame
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
