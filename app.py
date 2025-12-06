from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
import cv2
import numpy as np

def preprocess_digit(image_bytes):
    # Convert bytes to numpy
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 1. Remove noise using Gaussian blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # 2. Apply adaptive threshold to isolate digit
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    # 3. Find contours (digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    # largest contour = digit
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 4. Crop digit region
    digit = thresh[y:y+h, x:x+w]

    # 5. Resize digit while keeping aspect ratio
    height, width = digit.shape
    if height > width:
        factor = 20 / height
        height = 20
        width = int(width * factor)
    else:
        factor = 20 / width
        width = 20
        height = int(height * factor)

    digit = cv2.resize(digit, (width, height))

    # 6. Add padding to make 28×28
    padded = np.pad(
        digit,
        (
            ((28 - height) // 2, (28 - height) - (28 - height) // 2),
            ((28 - width) // 2, (28 - width) - (28 - width) // 2)
        ),
        "constant", constant_values=0
    )

    # 7. Normalize (0–1)
    norm = padded / 255.0

    # Return for model
    return norm.reshape(1, 28, 28), padded


# Load saved model
model = tf.keras.models.load_model("mnist_linear.h5")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img_bytes = file.read()

    processed, downloadable_img = preprocess_digit(img_bytes)

    # Save preprocessed image
    cv2.imwrite("preprocessed.png", downloadable_img)

    # Predict class
    pred = np.argmax(model.predict(processed))

    return jsonify({
        "prediction": int(pred)
    })

@app.route("/download_preprocessed")
def download_preprocessed():
    return send_file("preprocessed.png", as_attachment=True)

# Run flask app
app.run(port=5000)
