import cv2
import sys

# Load the face cascade
faceCascade = cv2.CascadeClassifier("E:\Chirag\python\haarcascade_frontalface_default.xml")

# Create a video capture object
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # For each face found, draw a rectangle around it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Quit if the user presses ESC
    if cv2.waitKey(1) == 27:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
