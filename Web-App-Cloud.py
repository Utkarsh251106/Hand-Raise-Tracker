import streamlit as st
import sys
import traceback

# Add error handling for imports
try:
    import cv2
    import numpy as np
    import mediapipe as mp
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(traceback.format_exc())
    sys.exit(1)

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
st.write("Upload a video to count hand raises")

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Reset button
reset = st.button("Reset Counter")

# Initialize counter in session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Reset counter if reset button is pressed
if reset:
    st.session_state.counter = 0

# Placeholder for video display and counter
frame_window = st.empty()
counter_display = st.empty()

# Process video if file is uploaded
if uploaded_file is not None:
    try:
        # Save uploaded file to a temporary location
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Open the video
        cap = cv2.VideoCapture(tfile.name)
        
        # Verify video capture
        if not cap.isOpened():
            st.error("Error: Could not open video file")
            sys.exit(1)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Progress bar
        progress_bar = st.progress(0)
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            previous_position = "Unknown"
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Current frame number and progress
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress_bar.progress(int(current_frame / total_frames * 100))
                
                # Flip and convert frame
                frame = cv2.flip(frame, 1)
                image_height, image_width, _ = frame.shape
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                results = pose.process(rgb_image)
                
                if results.pose_landmarks:
                    # Draw landmarks
                    mp_drawings.draw_landmarks(
                        rgb_image,
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                    
                    # Extract landmarks
                    landmarks = results.pose_landmarks.landmark
                    
                    # Calculate angles for left and right sides
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
                    
                    # Calculate angles
                    left_angle = calc_angle(left_shoulder, left_elbow, left_hip)
                    right_angle = calc_angle(right_shoulder, right_elbow, right_hip)
                    
                    # Count hand raises
                    if left_angle < 20 and right_angle < 20 and previous_position != "Up":
                        st.session_state.counter += 1
                    
                    previous_position = "Up" if left_angle < 20 and right_angle < 20 else "Not Up"
                
                # Display frame
                frame_window.image(rgb_image, channels="RGB")
                
                # Update counter
                counter_display.metric(label="Total Hand Raises", value=st.session_state.counter)
        
        # Close video capture
        cap.release()
        
        # Complete progress
        progress_bar.progress(100)
        
        # Final results
        st.success(f"Video processing complete. Total hand raises: {st.session_state.counter}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())
else:
    st.info("Please upload a video file to start")