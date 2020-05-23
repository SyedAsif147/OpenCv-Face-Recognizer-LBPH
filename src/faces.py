import numpy as np
import cv2
import pickle


count = 0

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pkl", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
	for x,y,w,h in faces:
		# print(x,y,w,h)
		roi_gray = gray[y:y+h,x:x+w]  # [ystart:yend , xstart:xend]
		roi_color = frame[y:y+h,x:x+w]

		# recognizer
		id_, conf = recognizer.predict(roi_gray)
		print(conf)
		if conf >= 70: # and conf <=85:
			print(id_)
			print(labels[id_])

			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (0, 255, 0)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		count += 1
		# image capturing:
		# img_item = "./image-sif/" + str(count) + '.jpg'
		img_item = 'img_save.png'

		color = (255, 0, 0) # BGR
		stroke = 2 
		end_cord_x = x + w
		end_cord_y = y + h
		face = cv2.resize(roi_color, (400, 400))
		cv2.imwrite(img_item, roi_color)
		# cv2.putText(face, , (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

		# subitems = smile_cascade.detectMultiScale(roi_gray)
		# for ex, ey, ew, eh in subitems:
		# 	cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'): # Press q to exit
		break

cap.release()
cv2.destroyAllWindows()