# 🤟 Sign Language Word Recognition from Video

This project implements a system for recognizing isolated words in sign language from short video clips, using MediaPipe for pose and hand landmark detection and a deep learning model (LSTM) for temporal classification.

## 🎯 Objective

> Recognize individual sign language words from videos using pose and hand keypoints as input to a temporal deep learning model.

The system is designed to support future applications in real-time gesture recognition, human-computer interaction, and assistive technologies for the hearing-impaired.

## 🛠️ Technologies Used

- **Python** for scripting and model development  
- **MediaPipe** for extracting pose and hand landmarks  
- **OpenCV** for video frame processing  
- **NumPy** for data preprocessing and time series construction  
- **TensorFlow (LSTM)** for building and training the classification model  
- **Matplotlib** for training visualization and evaluation

## 🔁 Pipeline Overview

1. **Data Collection & Preprocessing**
   - Video data collected for each sign word (user-recorded clips)
   - Extracted landmarks (pose, left/right hand) per frame using MediaPipe
   - Stored keypoints (x, y, z) as time-series data using NumPy arrays

2. **Feature Engineering**
   - Normalized keypoint coordinates to reduce variation between users
   - Stacked hand and pose keypoints per frame to create a consistent input structure
   - Constructed sequences of fixed length (30 frames per video)

3. **Modeling**
   - Built an LSTM-based model using TensorFlow/Keras
   - Input: sequence of keypoints  
   - Output: predicted sign word label
   - Used categorical cross-entropy loss and accuracy metrics

4. **Training & Evaluation**
   - Trained on a small custom dataset of sign language words
   - Visualized loss and accuracy curves using Matplotlib
   - Evaluated model with confusion matrix and classification accuracy

## 🚀 Future Improvements
- Expand vocabulary with more sign words  
- Improve model robustness with larger and more diverse datasets  
- Add real-time recognition via webcam stream  
- Integrate with GUI for accessibility applications
