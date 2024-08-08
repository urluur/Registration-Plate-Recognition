import cv2
import numpy as np
import imutils
import easyocr

reader = easyocr.Reader(['en'])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 200) # Edge Detection
    return edged

def extract_license_plates(image):
    preprocessed_image = preprocess_image(image)
    contours = cv2.findContours(preprocessed_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    license_plates = []

    for contour in contours:
        peri = cv2.arcLength(contour, True) # approx contour
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # If our approximated contour has four points, we assume we have found a license plate
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cropped = image[y:y + h, x:x + w]
            license_plates.append((cropped, approx))

    return license_plates

def main():
    # Initialize the webcam video stream
    cap = cv2.VideoCapture(0)
    cv2.waitKey(100)

    detected_plates = []

    while True:
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        
        license_plates = extract_license_plates(frame)

        for license_plate_image, license_plate_cnt in license_plates:
            # Use EasyOCR to recognize text
            result = reader.readtext(license_plate_image)
            for (bbox, text, prob) in result:
                if prob > 0.5 and text.strip() and text not in detected_plates:
                    detected_plates.append(text.strip())
                    print("Detected License Plate:", text.strip())
                    # Draw a rectangle around the license plate
                    cv2.drawContours(frame, [license_plate_cnt], -1, (0, 255, 0), 3)

        # Display the resulting frame
        cv2.imshow('Video Stream', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    # Print the list of detected plates
    print("Detected License Plates:", detected_plates)

if __name__ == "__main__":
    main()
