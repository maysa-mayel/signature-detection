from flask import Flask, request, jsonify
from pdf2image import convert_from_path
import pandas as pd
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os 


app = Flask(__name__)

model_path ="C:\Users\USER\Desktop\signture task\best (9).pt"
model=YOLO(model_path)
CSV_FILE = 'signature.csv'

def detect_and_crop(image_path, username):
    img = cv2.imread(image_path)
    results = model(image_path)
    boxes = results[0].boxes.xyxy.tolist()

     # Only save the first signature detected
    if not boxes:
        return None

    x1, y1, x2, y2 = boxes[0]  
    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
    
    # Convert cropped image to bytes
    _, buffer = cv2.imencode('.jpg', cropped_img)
    img_bytes = buffer.tobytes()

    return img_bytes



def save_to_csv(username, signature_image_bytes):
    df = pd.DataFrame([{'username': username, 'signature': signature_image_bytes}])

    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)



def process_file(file_path, username):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            img_path = f'page_{i}.jpg'
            page.save(img_path, 'JPEG')
            cropped_img_bytes = detect_and_crop(img_path, username)
            if cropped_img_bytes:
                save_to_csv(username, cropped_img_bytes)
            os.remove(img_path)
    else:
        cropped_img_bytes = detect_and_crop(file_path, username)
        if cropped_img_bytes:
            save_to_csv(username, cropped_img_bytes)


@app.route('/detect-signature', methods=['POST'])
def detect_signature():
    if 'file' not in request.files or 'username' not in request.form:
        return jsonify({'error': 'No file or username provided'}), 400
    
    file = request.files['file']
    username = request.form['username']
    
    # Save the uploaded file temporarily
    temp_file_path = f'temp_{file.filename}'
    file.save(temp_file_path)
    
    # Process the file
    process_file(temp_file_path, username)
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    
    return jsonify({'message': 'Signature detected and saved successfully'}), 200


if __name__ == '__main__':
    app.run(debug=True)
