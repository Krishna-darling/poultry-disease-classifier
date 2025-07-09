from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Mock prediction - Replace this with model prediction later
    prediction = "Healthy Bird 🐔"

    return render_template('predict.html', prediction=prediction, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
