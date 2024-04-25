import cv2
import numpy as np
from keras.models import load_model

# Loading pre-trained model
model = load_model('model_vgg16.h5')

# Defining capture region coordinates
roi_x_start = 50  # Left X coordinate of the region
roi_y_start = 50  # Top Y coordinate of the region
roi_width = 256   # Width of the region
roi_height = 256  # Height of the region

# Defining capture flag (initially False)
capture_flag = False

def capture_image(frame):
  global capture_flag  # Access global capture flag

  # Extract the region of interest (ROI)
  roi = frame[roi_y_start:roi_y_start+roi_height, roi_x_start:roi_x_start+roi_width]

  # Resize the ROI to the model's input size
  resized_roi = cv2.resize(roi, (64, 64))
  # Expand dimensions for model prediction (if required)
  resized_roi = np.expand_dims(resized_roi, axis=0)  # Adding a batch dimension

  
  result = model.predict(resized_roi)

  # Get the predicted class index


  if result[0][0] == 1:
    prediction = '0'
  elif result[0][1] == 1:
    prediction = '1'
  elif result[0][2] == 1:
    prediction = '2'
  elif result[0][3] == 1:
    prediction = '3'
  elif result[0][4] == 1:
    prediction = '4'
  elif result[0][5] == 1:
    prediction = '5'
  elif result[0][6] == 1:
    prediction = '6'
  elif result[0][7] == 1:
    prediction = '7'
  elif result[0][8] == 1:
    prediction = '8'
  elif result[0][9] == 1:
    prediction = '9'
  elif result[0][10] == 1:
    prediction = 'A'
  elif result[0][11] == 1:
    prediction = 'B'
  elif result[0][12] == 1:
    prediction = 'C'
  elif result[0][13] == 1:
    prediction = 'D'
  elif result[0][14] == 1:
    prediction = 'E'
  elif result[0][15] == 1:
    prediction = 'F'
  elif result[0][16] == 1:
    prediction = 'G'
  elif result[0][17] == 1:
    prediction = 'H'
  elif result[0][18] == 1:
    prediction = 'I'q
  elif result[0][19] == 1:
    prediction = 'J'
  elif result[0][20] == 1:
    prediction = 'K'
  elif result[0][21] == 1:
    prediction = 'L'
  elif result[0][22] == 1:
    prediction = 'M'
  elif result[0][23] == 1:
    prediction = 'N'
  elif result[0][24] == 1:
    prediction = 'O'
  elif result[0][25] == 1:
    prediction = 'P'
  elif result[0][26] == 1:
    prediction = 'Q'
  elif result[0][27] == 1:
    prediction = 'R'
  elif result[0][28] == 1:
    prediction = 'S'
  elif result[0][29] == 1:
    prediction = 'T'
  elif result[0][30] == 1:
    prediction = 'U'
  elif result[0][31] == 1:
    prediction = 'V'
  elif result[0][32] == 1:
    prediction = 'W'
  elif result[0][33] == 1:
    prediction = 'X'
  elif result[0][34] == 1:
    prediction = 'Y'
  elif result[0][35] == 1:
    prediction = 'Z'
  else:
    prediction = '  '
  # Display captured image, label, and confidence
  cv2.imshow('Captured Image', roi)
  #confidence = prediction[0][result]  # Assuming confidence is the max probability
  cv2.putText(frame, f"Label: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  #cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  print(prediction)
  # Reset capture flag
  capture_flag = False


  
def on_mouse_click(event, x, y, flags, param):
  global capture_flag  # Access global capture flag

  if event == cv2.EVENT_LBUTTONDOWN:
    if roi_x_start <= x <= roi_x_start + roi_width and roi_y_start <= y <= roi_y_start + roi_height:
      capture_flag = True  # Set capture flag on click within ROI

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change index for other cameras

# Create window for video display
cv2.namedWindow('Webcam Feed')
cv2.setMouseCallback('Webcam Feed', on_mouse_click)  # Set mouse click callback

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Draw the region of interest (ROI) bounding box
  cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_start + roi_width, roi_y_start + roi_height), (0, 255, 0), 2)

  # Capture image if capture flag is set
  if capture_flag:
    capture_image(frame.copy())  # Use a copy to avoid modifying the original frame

  # Display the webcam feed
  cv2.imshow('Webcam Feed', frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()
