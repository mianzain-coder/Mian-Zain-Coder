import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import cv2

# Initialize Flask
app = Flask(__name__)

# Where uploads/outputs are stored
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Face detector (Haar)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Full-body person detector (HOG)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------- ROUTES ----------
@app.route('/', methods=['GET'])
def index():
    # Render your main HTML template
    return render_template('human_index.html', result=None, image_url=None)


@app.route('/predict', methods=['POST'])
def predict():
    # Validate upload
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Unique name for uploaded file
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    in_name = f"upload_{ts}.jpg"
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], in_name)

    # Save original upload
    img = Image.open(file.stream).convert('RGB')
    img.save(in_path, format='JPEG')

    # Read with OpenCV
    cv_img = cv2.imread(in_path)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 1) Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green

    # 2) Detect full-body humans
    rects, weights = hog.detectMultiScale(
        cv_img, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    for (x, y, w, h) in rects:
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

    # Result message
    total_humans = len(faces) + len(rects)
    if total_humans == 0:
        result = "No human detected."
    elif total_humans == 1:
        result = "1 human detected."
    else:
        result = f"{total_humans} humans detected."

    # Save annotated image
    out_name = f"annotated_{ts}.jpg"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    cv2.imwrite(out_path, cv_img)

    # Render page with result + image
    return render_template(
        'human_index.html',
        result=result,
        image_url=url_for('static', filename=f'uploads/{out_name}')
    )


if __name__ == '__main__':
    # Starts the development web server so users can access it in a browser
    app.run(debug=True)
