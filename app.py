from flask import Flask, request, send_file, render_template
import os
import subprocess
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_IMAGE = "combined_result.jpg"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Call your C++ program
            subprocess.run([
                "./sudoku_ocr", filepath, "digit_classifier.onnx", "0.7"
            ])

            return send_file(RESULT_IMAGE, mimetype='image/jpeg')

    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
