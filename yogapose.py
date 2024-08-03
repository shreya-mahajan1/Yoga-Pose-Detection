# %%
# !pip install mediapipe opencv-python

# %%
#IMPORTING ALL THE NECESSARY LIBRARIES
import os
import math
import numpy as np
import mediapipe as mp
import cv2
from time import time
import matplotlib.pyplot as plt

# %%
#LOADING AND PRINTING IMAGE

#solutions is a module in mp which has built in solutions for various computer vision tasks
#solutions.pose is a built in solution which detects a persons pose by identifying key points such as joints and body parts 
mp_pose=mp.solutions.pose
#Pose is a class in mp

#static_image_mode="True" means that we are working on an image and not a video, default value is false
#latency- delay between when a user takes an action on a network when they get a response

#min_detection_confidence= It is the minimum tracking confidence ([0.0, 1.0]) required to consider the landmark-tracking model's tracked pose landmarks valid. Detected poses with confidence below this threshold will be ignored, so increasing its value increases the robustness, but also increases the latency. Its default value is 0.5.

# model_complexity=2 ,there are three possible values: 0, 1, or 2. The higher the value, the more accurate the results are, but at the expense of higher latency. Its default value is 1.

pose=mp_pose.Pose(static_image_mode="True",min_detection_confidence=0.3,model_complexity=2)

#mp.solutions.drawing_utils:for annotating images or video frames with landmarks, bounding boxes, and other visualizations.
mp_drawing=mp.solutions.drawing_utils
#loading he image by specifying path
img_path=r'C:\Users\mridu\OneDrive\Desktop\media\cobrapose.jpg'
#imread function from opencv reads image from specified path and stores it in a variable for further processing
sample_img=cv2.imread(img_path)
#check if image is loaded successfully
if sample_img is None:
    print("error in loading the image")
else:
    #displaying the image
    plt.title("sample image") #giving tile to plot
    plt.axis('off') #done to remove coordinates 
    plt.imshow(sample_img[:, :, ::-1])#converts BGR(used by openCV) to RGB(used by matplotlib) by reversing color channels
    plt.show()

# %% [markdown]
# ####################### LANDMARK POINTS #######################

# %%
#DISPLAYS LANDMARKS


#Applies the pose detection model to the provided image and returns the results. results contains information about the detected pose, including landmark locations.

#there are total 32 landmark posititons:
'''
0 - Nose 1 - Left eye inner 2 - Left eye 3 - Left eye outer 4 - Right eye inner 5 - Right eye 6 - Right eye outer 7 - Left ear 8 - Right ear
9 - Mouth left 10 - Mouth right 11 - Left shoulder 12 - Right shoulder 13 - Left elbow 14 - Right elbow 15 - Left wrist 16 - Right wrist
17 - Left pinky finger 18 - Right pinky finger 19 - Left index finger 20 - Right index finger 21 - Left thumb 22 - Right thumb 23 - Left hip
24 - Right hip 25 - Left knee 26 - Right knee 27 - Left ankle 28 - Right ankle 29 - Left heel 30 - Right heel 31 - Left foot index
32 - Right foot index'''

results=pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
#converts bgr to rgb as pose.process expects img in rgb format(redundant here as it is already done so can simply write :)
# results = pose.process(sample_img)
if results.pose_landmarks: #checks if any landmark points are found
    for i in range (2): #prints first 2 landmark points i.e nose an left eye inner
        print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')
        #prints landmark points name its x,y, and z coordinate as well as its visiblity

# %%
# DISPLAYS LANDMARKS IN ORIGINAL SCALE

#sample.img.shape returns a tuple of 3 values img height,width and channels(colours) _is used to catch channels and is ignored
image_height,image_width,_=sample_img.shape
if results.pose_landmarks: #checks if any landmark points are found
    for i in range (2): #prints first 2 landmark points i.e nose an left eye inner
        print(f'{mp_pose.PoseLandmark(i).name}:')
        #Prints the x-coordinate of the landmark in the original scale (pixel coordinates) by multiplying the normalized x-coordinate with the image width.
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        #Prints the y-coordinate of the landmark in the original scale (pixel coordinates) by multiplying the normalized y-coordinate with the image height.
        print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_height}')
        #Prints the z-coordinate of the landmark in the original scale (pixel coordinates) by multiplying the normalized z-coordinate with the image width.
        print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
        print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')#prints visibility

# %%
#ANNOTATING THE IMAGE AND DRAWING LANDMARKS 

#creates a copy of original img to avoid making changes in it
img_copy=sample_img.copy()
#drawing landmarks
mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
#specifying size of figure
fig=plt.figure(figsize=[10,10])
#displaying img
plt.title("output")
plt.axis('Off')
plt.imshow((img_copy[:, :, ::-1]))#converts bgr to rgb
plt.show()

# %%
# Plot Pose landmarks in 3D.
# POSE_WORLD_LANDMARKS that is another list of pose landmarks in world coordinates that has the 3D coordinates in meters with the origin at the center between the hips of the person.
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# %%
# Detecting Pose 
def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121)
        plt.imshow(image[:,:,::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis('off')
    
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks

# %%
#TESTING CODE WITH SOME EXAMPLES

image = cv2.imread(r'C:\Users\mridu\OneDrive\Desktop\media\warriorIIpose.jpg')
detectPose(image, pose, display=True)
image = cv2.imread(r'C:\Users\mridu\OneDrive\Desktop\media\sample2.jpg')
detectPose(image, pose, display=True)

# %%
#CLACULATING ANGLE

def calculateAngle(landmark1 ,landmark2, landmark3):
    x1,y1,_=landmark1
    x2,y2,_=landmark2
    x3,y3,_=landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if(angle<0):
        angle+=360
    return angle

# %%
#TESTING ABOVE CODE

# Calculate the angle between the three landmarks.
angle = calculateAngle((558, 326, 0), (642, 333, 0), (718, 321, 0))
# Display the calculated angle.
print(f'The calculated angle is {angle}')

# %% [markdown]
# ####################### ANGLES #######################

# %%
def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label

# %%
# Read a sample image and perform pose classification on it.
image = cv2.imread(r'C:\Users\mridu\OneDrive\Desktop\media\warriorIIpose.jpg')
output_image, landmarks = detectPose(image, pose, display=False)
if landmarks:
    classifyPose(landmarks, output_image, display=True)
image = cv2.imread(r'C:\Users\mridu\OneDrive\Desktop\media\Tpose1.jpg')
output_image, landmarks = detectPose(image, pose, display=False)
if landmarks:
    classifyPose(landmarks, output_image, display=True)
image = cv2.imread(r'C:\Users\mridu\OneDrive\Desktop\media\treepose.jpg')
output_image, landmarks = detectPose(image, pose, display=False)
if landmarks:
    classifyPose(landmarks, output_image, display=True)
image = cv2.imread(r'C:\Users\mridu\OneDrive\Desktop\media\sample5.jpg')
output_image, landmarks = detectPose(image, pose, display=False)
if landmarks:
    classifyPose(landmarks, output_image, display=True)

# %%
#DETECTION OF POSES IN REAL-TIME WEBCAM FEED/VIDEO

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(1)

# Create named window for resizing purposes
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)


# Initialize the VideoCapture object to read from a video stored in the disk.
#video = cv2.VideoCapture('media/running.mp4')

# Set video camera size
video.set(3,1280)
video.set(4,960)

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Iterate until the video is accessed successfully.
while video.isOpened():
    
    # Read a frame.
    ok, frame = video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Break the loop.
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, _ = detectPose(frame, pose_video, display=False)
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
    if (time2 - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2
    
    # Display the frame.
    cv2.imshow('Pose Detection', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()

# %%
#CLASSIFICATION OF POSES IN REAL-TIME WEBCAM FEED/VIDEO

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
    if landmarks:
        
        # Perform the Pose Classification.
        frame, _ = classifyPose(landmarks, frame, display=False)
    
    # Display the frame.
    cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()


