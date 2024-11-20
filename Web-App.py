import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import platform

# Initialize MediaPipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawings = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    cb = c - b
    
    dot_product = np.dot(ab, cb)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_cb = np.linalg.norm(cb)
    
    cos_angle = dot_product / (magnitude_ab * magnitude_cb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)
    
    return angle

# Streamlit app layout
st.title("Hand Raise Counter")

# Initialize session state variables
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'previous_position' not in st.session_state:
    st.session_state.previous_position = "Unknown"
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Camera capture function
def get_camera_capture():
    capture_methods = [
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (0, "Default Camera"),
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation")
    ]
    
    # Filter methods based on operating system
    if platform.system() == "Linux":
        capture_methods = [(cv2.CAP_V4L2, "V4L2 (Linux)"), (0, "Default Camera")]
    elif platform.system() == "Windows":
        capture_methods = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Microsoft Media Foundation"), (0, "Default Camera")]
    
    for method, name in capture_methods:
        try:
            cap = cv2.VideoCapture(0, method)
            if cap.isOpened():
                return cap
        except Exception as e:
            st.error(f"Error with {name} method: {e}")
    
    st.error("Could not open camera with any method")
    return None

# Toggle camera function
def toggle_camera():
    if not st.session_state.camera_running:
        # Start camera
        st.session_state.cap = get_camera_capture()
        st.session_state.camera_running = True
    else:
        # Stop camera
        if st.session_state.cap:
            st.session_state.cap.release()
        st.session_state.camera_running = False
        st.session_state.cap = None

# Create a single camera control button
st.button('Start/Stop Camera', on_click=toggle_camera)

# Display frame and counter placeholders
FRAME_WINDOW = st.empty()
counter_display = st.empty()

# Main camera and pose detection loop
if st.session_state.camera_running and st.session_state.cap is not None:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while st.session_state.camera_running:
            success, frame = st.session_state.cap.read()
            if not success:
                st.warning("Failed to capture video.")
                break

            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            results = pose.process(rgb_image)
            if results.pose_landmarks:
                mp_drawings.draw_landmarks(
                    rgb_image,  # Draw on the RGB image
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                landmarks = results.pose_landmarks.landmark
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width, 
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width, 
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width, 
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
                
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width, 
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width, 
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height]
                
                left_angle = calc_angle(left_shoulder, left_elbow, left_hip)
                right_angle = calc_angle(right_shoulder, right_elbow, right_hip)
                
                if left_angle < 20 and right_angle < 20 and st.session_state.previous_position != "Up":
                    st.session_state.counter += 1
                
                st.session_state.previous_position = "Up" if left_angle < 20 and right_angle < 20 else "Not Up"
            
            # Display the frame in Streamlit
            FRAME_WINDOW.image(rgb_image)  # Ensure RGB image is passed

            # Update the counter display
            counter_display.metric(label="Counter", value=st.session_state.counter)

# Cleanup on app exit or rerun
if st.session_state.cap:
    st.session_state.cap.release()