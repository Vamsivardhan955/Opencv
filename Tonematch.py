import cv2
import numpy as np

def detect_skin_and_tone(image):
    """
    Detects skin regions in the given image and classifies skin tone.
    
    Args:
        image (numpy.ndarray): Input BGR image.
    
    Returns:
        tuple: Skin detection result, binary mask, and skin tone category.
    """
    # Convert the image to YCrCb color space
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define the skin color range in YCrCb
    lower_bound = np.array([0, 130, 75], dtype=np.uint8)  # Adjusted lower bound of skin
    upper_bound = np.array([255, 180, 135], dtype=np.uint8)  # Adjusted upper bound of skin
    
    # Create a binary mask for skin color
    skin_mask = cv2.inRange(ycrcb_image, lower_bound, upper_bound)
    
    # Apply the mask to extract skin regions
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    
    # Analyze the Y (luminance) channel to classify skin tone
    y_channel = ycrcb_image[:, :, 0]  # Extract the Y (luminance) channel
    masked_y = cv2.bitwise_and(y_channel, y_channel, mask=skin_mask)  # Mask Y channel
    non_zero_y = masked_y[masked_y > 0]  # Consider only non-zero values in the masked Y
    
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
    
    # Wait for the user to press 'q' to capture one frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access the webcam.")
            break
        
        # Display the live video feed
        cv2.imshow("Live Video", frame)
        
        # Capture one frame when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Detect skin and tone
            skin, _, avg_brightness, skin_tone = detect_skin_and_tone(frame)
            
            # Print the skin tone
            print(f"Skin Colour: {skin_tone}")
            
            # Suggest shirt colors based on skin tone
            shirt_colors = recommend_shirt_colors(skin_tone)
            print(f"Recommended Shirt Colors: {', '.join(shirt_colors)}")
            
            # Display the results on captured frame
            cv2.putText(frame, f"Skin Tone: {skin_tone}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Captured Frame - Skin Detection", skin)
            
            break  # Exit the loop after capturing one frame
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
