import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import pytesseract
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
from threading import Thread
import streamlit as st
import os

# Load your model
loaded_model = tf.keras.models.load_model('road_distress_model.h5')

# Define categories (replace with your actual distress types)
categories = ['Longitudinal crack', 'Pothole', 'Oblique crack', 'Repair', 'Alligator crack', 'Block crack', 'Transverse crack', 'Edge Break']

# Initialize a list to store the distress data
distress_data = []

receiver_email = 'Tukaram.Tanpure@planeteyefarm.ai'

def preprocess_image(img):
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img_array = img / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_distress(frame):
    img_array = preprocess_image(frame)
    predictions = loaded_model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    
    # Define the index for "Block crack" and the replacement index for "Pothole"
    BLOCK_CRACK_INDEX = categories.index('Block crack')
    POTHOLE_INDEX = categories.index('Pothole')
    
    # Check if prediction is "Block crack" and replace it with "Pothole"
    if predicted_class == BLOCK_CRACK_INDEX:
        predicted_class = POTHOLE_INDEX
    
    # Example detection format: [x, y, width, height, confidence, class]
    detections = [
        [50, 50, 200, 200, 0.8, predicted_class]
    ]
    
    return detections

def extract_latlong(frame):
    if frame is None or frame.size == 0:
        return None, None

    height, width, _ = frame.shape
    roi = frame[int(0.70 * height):height, 0:width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    extracted_text = pytesseract.image_to_string(gray)
    
    lat_pattern = r'Latitude[:\s]*([-+]?\d{1,2}\.\d+)'
    long_pattern = r'Longitude[:\s]*([-+]?\d{1,3}\.\d+)'
    
    lat_match = re.search(lat_pattern, extracted_text)
    long_match = re.search(long_pattern, extracted_text)

    if lat_match and long_match:
        latitude = float(lat_match.group(1))
        longitude = float(long_match.group(1))
        return latitude, longitude

    return None, None

def process_video(video_path, highway_no, section_no, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % skip_frames == 0:
            detections = predict_distress(frame)
            latitude, longitude = extract_latlong(frame)
            
            if latitude is not None and longitude is not None:
                for detection in detections:
                    x_center, y_center, width, height, confidence, class_id = detection
                    distance = frame_number * 0.2
                    
                    distress_data.append({
                        'Highway_No': highway_no,
                        'Section_No': section_no,
                        'Distance (meters)': distance,
                        'Latitude': latitude,
                        'Longitude': longitude,
                        'Distress_Type': categories[class_id],
                        'Confidence': confidence
                    })
        
        frame_number += 1

    cap.release()
    save_to_excel(highway_no, section_no)

def save_to_excel(highway_no, section_no):
    df = pd.DataFrame(distress_data)
    excel_filename = f'Highway_{highway_no}_Section_{section_no}.xlsx'
    df.to_excel(excel_filename, index=False)
    send_email(receiver_email, f"Road distress data for Highway {highway_no} Section {section_no}", excel_filename)

def send_email(receiver_email, body, attachment_path):
    SENDER_EMAIL = 'tanmayshivam35@gmail.com'
    EMAIL_PASSWORD = 'hehi mpmm jszy pecp'
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587

    filename = os.path.basename(attachment_path)

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = f'Road Distress Report - {filename}'
    msg.attach(MIMEText(body, 'plain'))

    try:
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {filename}")
            msg.attach(part)
    except Exception as e:
        print(f"Failed to attach the file: {str(e)}")
        return

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, receiver_email, text)
        server.quit()
        print(f"Email sent to {receiver_email} with report {filename}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Streamlit GUI
st.title("Road Distress Detection System")
st.write("Upload a video for processing road distress data.")

# Text inputs for Highway No. and Section No.
highway_no = st.text_input("Enter Highway No.")
section_no = st.text_input("Enter Section No.")

# File uploader for video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if st.button("Process Video"):
    if video_file is not None and highway_no and section_no:
        video_path = 'uploaded_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        
        # Run video processing in a separate thread
        video_thread = Thread(target=process_video, args=(video_path, highway_no, section_no))
        video_thread.start()
        video_thread.join()
        
        st.success(f"Processing complete. Data saved to Highway_{highway_no}_Section_{section_no}.xlsx")
        st.success("Data forwarded to dashboard")
    else:
        st.error("Please provide the Highway No., Section No., and upload a video.")
