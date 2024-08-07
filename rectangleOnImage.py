### /////////////////////////////////////////////////////////
### DRAWING RECTANGLE ON THE IMAGE - for now not relavant
### /////////////////////////////////////////////////////////

# image_path = './archive/images/Cars175.png'
# image = cv2.imread(image_path)

# # Read from the dictionary
# extracted = dataset['Cars175.png']

# # Get the values needed to draw a rectangle
# x = extracted['xmin']
# x2 = extracted['xmax']
# y = extracted['ymin']
# y2 = extracted['ymax']
# # Draw a rectangle 
# cv2.rectangle(image, (x,y), (x2, y2), (255, 0 , 0), 2)

# # Cuts the image, makes it black and white and tries to read the text
# img2 = image[y-5:y2+5, x-5:x2+5]
# grey_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# text = pytesseract.image_to_string(grey_img)

# # Draw a text next to the recvtangle including registration
# cv2.putText(image, text, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# # Displays the image in a new window
# cv2.imshow('Test', image);
# cv2.waitKey(0)
# cv2.destroyAllWindows()
