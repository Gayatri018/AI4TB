from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your TB detection model
model = load_model("./models/tb_detection_model.h5")  # Update path if needed

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image (assuming RGB input, 224x224 size, normalized)
    image = Image.open(filepath).resize((224, 224)).convert('RGB')
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Determine output type
    if prediction.shape[-1] == 1:  # Binary classification with sigmoid
        result = float(prediction[0][0])
        predicted_label = 1 if result >= 0.5 else 0
    else:  # Softmax with 2 or more classes
        predicted_label = int(np.argmax(prediction[0]))
        result = prediction[0][predicted_label]

    return jsonify({
        'prediction': predicted_label,
        # 'confidence': f"{result:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)
