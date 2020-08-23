# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
import sys
# import the necessary packages
from datetime import datetime
import pytz
#print(datetime.now(pytz.timezone('Asia/Kolkata')).time())
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import re
import speech_recognition as sr 
from gtts import gTTS   
import os 
ty=0

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 70.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 24.0

# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("found.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH





mytext = 'Please Speak now'
  
# Language in which you want to convert 
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)

file="objdet.mp3"
myobj.save("objdet.mp3")

mic_name = "USB2.0 PC CAMERA: Audio (hw:1,0)"

sample_rate = 48000
chunk_size = 2048
r = sr.Recognizer() 
  
mic_list = sr.Microphone.list_microphone_names() 
  
for i, microphone_name in enumerate(mic_list): 
    if microphone_name == mic_name: 
        device_id = i 
  
with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                        chunk_size = chunk_size) as source: 
    r.adjust_for_ambient_noise(source) 
    # Playing the converted file
    
    
    os.system("mpg123 " + file)
    print ("Say Something")
    audio = r.listen(source) 
          
    try: 
        text = r.recognize_google(audio) 
        print ("you said: " + text) 
      
      
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio") 
      
	
if(re.search("time",text)):
	mytext=str(datetime.now(pytz.timezone('Asia/Kolkata')).time())
	language = 'en'
	myobj = gTTS(text=mytext, lang=language, slow=False)
	file="objdet.mp3"
	myobj.save("objdet.mp3")
	os.system("mpg123 " + file)
	sys.exit()
	








ct=0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"glass", "bus", "car", "cat", "seat", "cow", "table",
	"dog", "horse", "motorbike", "mum", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


'''
Create a class for obstacles
'''
class_obstruct=["chair","diningtable","sofa"]



obst_det=False #detecting any obstacle
obst_in_path=False #check if obstacle is in the path 

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			#print(label)
			x=label.split(":")
				
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			for i in x:
				k=''.join(i)
				if(k==text):
					lk=(startX+endX)/2
					rk=(startY+endY)/2
					ct=1
					if(lk>300):
						ty=1
					elif (lk>180 and lk <=300):
						ty=2
					elif (lk<=180):
						ty=3
					cv2.imwrite('found2'+'.jpg',frame)
					image = cv2.imread('found2.jpg')
					marker = find_marker(image)
					inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
					
			
			if x[0] in class_obstruct and x[0]!=text:
				obst_det=True
				lobst_det=(startX+endX)/2
				robst_det=(startY+endY)/2
				if(lobst_det>300):
					obst_ty=1
				elif lobst_det>180 and lobst_det<=300:
					obst_ty=2
				elif lobst_det<=180:
					obst_ty=3	
						
			if ct==1 and obst_det==True:
				if obst_ty==ty:
					obst_in_path=True
				else:
					obst_in_path=False
			
				
					
							
        	
  
# Language in which you want to convert 
        
	
		
		
		
		
	
		
	
	
	
	if (ty==1):
		mytext='Object was found and is to the right'
	elif (ty==2):
		mytext='Object was found and is in the center'
	elif (ty==3):
		mytext='Object was found and is to the left'					
				
	language = 'en'
	myobj = gTTS(text=mytext, lang=language, slow=False)
	file="objdet.mp3"
	myobj.save("objdet.mp3")
				
					
            
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("You quit")
		break
	elif ct==1:
		if obst_in_path==False:
			os.system("mpg123 " + file)
			print("Object was found")
			print('distance is '+str(inches/12))
		else:
			mytext+=" Also obstacle ahead"
			myobj = gTTS(text=mytext, lang=language, slow=False)
			file="objdet.mp3"
			myobj.save("objdet.mp3")
			os.system("mpg123 " + file)
			print("obstacle in path")
			
			
		#print(lk)
		#print(rk)
		
		
		break	

	# update the FPS counter
	fps.update()






# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
