import torch
import torchvision.transforms as transforms
import cv2
import os
from collections import deque, Counter
from PIL import Image

# Imports from Transpondancer itself.
from model import CNN
import datahandler as dh # For preprocessing images

# Loading the pre-trained model
model = CNN()
# Make sure the pretrained model is in the parent directory (see README.md for Google Drive link)
pretrained_model_path = "./Transpondancer/models/ballet_retrained_retire_100epochs_79p.pth"
model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
model.eval()

# The classified poses
classes = ['Arabesque', 'Grand_Plié', 'Retiré']

# Function to classify a single image
def classify_image(image_path):
    # Preprocess the image.
    img_tensor = dh.preprocess_inference(image_path)
    
    # Add batch dimension => shape (1, 1, 90, 160)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        print(output)
        _, predicted = torch.max(output, 1)
    
    return classes[predicted.item()]

def classify_video(video_path, output_folder=None, frame_skip=5, window_size=7, confidence_threshold=0.45):
    """
    Classify poses in a video using frame-based CNN with smoothing over time and raw prediction debugging.
    """

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prediction_window = deque(maxlen=window_size)

    previous_prediction = None
    prediction_start_frame = None

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Convert frame to PIL and classify
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save('temp_frame.jpg')

            img_tensor = dh.preprocess_inference('temp_frame.jpg').unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                confidence = confidence.item()
                predicted_label = classes[predicted.item()]

                # Raw prediction always shown for debugging
                print(f'Raw frame {frame_idx}: {predicted_label} ({confidence:.2f})')

                # Apply confidence threshold
                if confidence < confidence_threshold:
                    prediction = 'none'
                else:
                    prediction = predicted_label

            # Add to smoothing window
            prediction_window.append(prediction)

            # When we have a full window, smooth and check for change
            if len(prediction_window) == window_size:
                smoothed_prediction = Counter(prediction_window).most_common(1)[0][0]

                if smoothed_prediction != previous_prediction:
                    if previous_prediction is not None and prediction_start_frame is not None:
                        end_frame = frame_idx
                        print(f'{prediction_start_frame}–{end_frame}: {previous_prediction}')
                    prediction_start_frame = frame_idx
                    previous_prediction = smoothed_prediction

        frame_idx += 1

    # Final block
    if previous_prediction is not None and prediction_start_frame is not None:
        print(f'{prediction_start_frame}-{frame_idx}: {previous_prediction}')

    cap.release()
    print('Video classification complete.')

# ------------------------------------------------------

def classify_video_for_reasoner(video_path, output_csv_path, frame_skip=5, confidence_threshold=0.45):
    """
    Classify a video and save frame-by-frame classifications to CSV for the ontology reasoner
    """
    import pandas as pd
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    # Initialize lists to store data
    frames = []
    poses = []
    confidences = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if (not frame_skip) or (frame_idx % frame_skip == 0):
            # Convert frame and classify
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save('temp_frame.jpg')
            
            img_tensor = dh.preprocess_inference('temp_frame.jpg').unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                confidence = confidence.item()
                predicted_label = classes[predicted.item()]
                
                # Store the results
                frames.append(frame_idx)
                poses.append(predicted_label.lower())  # Convert to lowercase for consistency
                confidences.append(confidence)
                
                print(f'Processed frame {frame_idx}: {predicted_label} ({confidence:.2f})')
        
        frame_idx += 1
    
    cap.release()
    
    # Create and save DataFrame
    results_df = pd.DataFrame({
        'frame': frames,
        'pose': poses,
        'confidence': confidences
    })
    
    results_df.to_csv(output_csv_path, index=False)
    print(f'Saved pose classifications to {output_csv_path}')
    return results_df

if __name__ == "__main__":
    # Example usage:
    video_path = "/Users/alisaavdic/Desktop/THESIS/l2d_eval/ballet/I_17/I_17.mp4"
    # output_folder = "classified_frames" # Optional
    classify_video(video_path) # output_folder=None, frame_skip=5, window_size=7, confidence_threshold=0.45)
    
    # Example usage for the reasoner:
    # video_path = "/Users/alisaavdic/Desktop/THESIS/l2d_eval/ballet/I_17/I_17.mp4"
    # output_csv = "/Users/alisaavdic/Desktop/THESIS/output/pose_classifications.csv"
    # classify_video_for_reasoner(video_path, output_csv)

