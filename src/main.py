import numpy as np
import cv2
import time
import argparse



# Init
from Face import CFace
from Marker import Marker

face_cascade_name = 'data/haarcascade_frontalface_default.xml'
eyes_cascade_name = 'data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# Loading the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('Error loading eyes cascade')
    exit(0)

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    CFaces = []
    for (x, y, w, h) in faces:
        CFaces.append(CFace((x, y, w, h)))

    Marker.MarkFaces(frame, CFaces)
    #
    # for (x,y,w,h) in faces:
    #     center = (x + w//2, y + h//2)
    #     frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    #
    #     faceROI = frame_gray[y:y+h,x:x+w]
    #     # In each face, detect eyes
    #     eyes = eyes_cascade.detectMultiScale(faceROI)
    #     for (x2,y2,w2,h2) in eyes:
    #         eye_center = (x + x2 + w2//2, y + y2 + h2//2)
    #         radius = int(round((w2 + h2)*0.25))
    #         frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    return frame

def normal_stream():
    cap = cv2.VideoCapture(0)  
    frames = 0
    # Start time measurement
    start = time.time()
    while(True):
        ret, frame = cap.read()        
        frames += 1
        # Display the frames
        cv2.imshow('frame',frame)
        # End time measurement
        end = time.time()
        elapsed_time = end - start
        # Wait for a second and print the number of frames 
        if elapsed_time >= 1:
            print (frames, "fps")
            frames = 0
            start = end
        
        # Exiting the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


def detect_good_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    return raw_corners;

def show_good_features(img, raw_corners):
    corners = np.int0(raw_corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    return img

def optical_flow(frame, old_frame, corners):

    color = np.random.randint(0, 255, (100, 3))
    mask = np.zeros_like(old_frame)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = corners[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    return img

def moving_good_features(frame, old_frame, corners):
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    return good_new.reshape(-1,1,2)


#This is the submission for assignment 2.3.2
# The optical flow displacement vectors are not shown
def face_detection_stream_moving_features():
    cap = cv2.VideoCapture(0)
    frames_in_second = 0
    total_frames = 0
    corners = 0
    # Start the time measurement
    start = time.time()
    ret, old_frame = cap.read()
    corners = detect_good_features(old_frame)
    while (True):
        ret, frame = cap.read()
        frames_in_second += 1
        total_frames += 1
        # Run the cascade face detector
        frame = detectAndDisplay(frame)
        if (total_frames % 300 == 0):
            print("300")
            corners = detect_good_features(frame)
        corners = moving_good_features(frame, old_frame, corners)
        frame = show_good_features(frame, corners)
        cv2.imshow('Capture - Face detection', frame)
        old_frame = frame
        # End the time measurement
        end = time.time()
        elapsed_time = end - start
        # Wait for a second and print the number of frames
        if elapsed_time >= 1:
            print(frames_in_second, "fps face detection")
            frames_in_second = 0
            start = end

        # Press Q to exit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# This combines the submission for 2.1, 2.2 and 2.3.1:
# We extract good features every 300 frames
# We draw them every frame
# The optical flow displacement vectors are calculated and visualised
def face_detection_stream():
    cap = cv2.VideoCapture(0)
    frames_in_second = 0
    total_frames = 0
    corners = 0
    # Start the time measurement
    start = time.time()
    ret, old_frame = cap.read()
    corners = detect_good_features(old_frame)
    while(True):
        ret, frame = cap.read()
        frames_in_second += 1
        total_frames += 1
        # Run the cascade face detector
        frame = detectAndDisplay(frame)
        if(total_frames % 300 == 0):
            print("300")
            corners = detect_good_features(frame)
        frame = show_good_features(frame, corners)
        frame = optical_flow(frame, old_frame, corners)
        cv2.imshow('Capture - Face detection', frame)
        old_frame = frame
        # End the time measurement
        end = time.time()
        elapsed_time = end - start
        # Wait for a second and print the number of frames 
        if elapsed_time >= 1:
            print (frames_in_second, "fps face detection")
            frames_in_second = 0
            start = end

        # Press Q to exit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
face_detection_stream()
face_detection_stream_moving_features()
#normal_stream()
