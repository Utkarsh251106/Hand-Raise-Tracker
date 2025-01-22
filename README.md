## Description
Hand Raise Tracker is a pose estimation detection project which counts how many times both the hands are raised straight up in the air simultaneously. It works by first getting all the necessary points which are to be used for calculation using Mediapipe library. After that it access camera using OpenCV to find and estimate the pose. Then the calculation of the angle between pelvic region, shoulder and elbow is done using NumPy library and a tracker is used to count the number of hand raises.

# How to run it?
### Step 1: Clone the Repository:
  
```bash
git clone https://github.com/Utkarsh251106/Hand-Raise-Tracker/tree/Second
```
### Step 2: Create a conda environment:
  
```bash
conda create -n venv python=3.12.7 -y
conda activate venv
```

### Step 3: Install the requirements:
  
```bash
pip install -r requirements.txt
```
### Step 4: To run the code or web-app:
  To run the code
```bash
# Start the Jupyter Notebook environment using the command
jupyter notebook
```
#### Run your Code.ipynb file


To run the Web-App
```bash
# Finally run the following command(Upload a video)
streamlit run Web-App-Cloud.py

# Finally run the following command(For Real Time)
streamlit run Web-App.py
```
### If you choose to upload the video, this is what should see

<img width="781" alt="Upload output" src="https://github.com/user-attachments/assets/b9f7b1e4-4f06-4b23-8f08-c389bcd0302f" />
