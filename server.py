from flask import Flask, render_template, request, jsonify
from os import path, environ
from werkzeug.utils import secure_filename
from parser import extract_troubleshooting

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = path.join('uploads', filename)
        file.save(filepath)

        try:
            response = extract_troubleshooting(filepath)
            return jsonify({"message": response})
        except Exception as e:
            print(e)
            return jsonify({"error": "Internal Server Error"}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 422


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    port = int(environ.get('PORT', 8081))
    app.run(debug=True, host='0.0.0.0', port=port)
