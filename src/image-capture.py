import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, scaleFactor = 1.5, minNeighbors=5)
    
    if faces is ():
    	return None
    
    # Crop all faces found
    for x,y,w,h in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = '../opencv-project/Datasets/train/folder' + str(count) + '.jpg'
        file_name = 'Image-capture.png'
        cv2.imwrite(file_name, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press q to exit
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
