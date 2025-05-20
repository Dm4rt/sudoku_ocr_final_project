# CS493-Project Final
STUDENT NAME: DANIEL MARTINEZ

## Sudoku OCR Detector

This project performs automatic Sudoku solving using computer vision and digit recognition. It uses:

1) OpenCV (C++)** to detect and extract the Sudoku grid
2) ONNX-trained CNN for digit classification
3) Flask to provide a simple web interface for user uploads
4) This project uses the MNIST dataset to train the digit classification model.


-----------------------------------------
How It Works

1) A User can upload a Sudoku image through the web interface.
2. The C++ program will:
   - Detect and warp the Sudoku board using OpenCV.
   - Segment the board into 81 different cells (all stored in the cells folder fyi).
   - Uses an ONNX-trained neural net to classify the digits.
   - Solves the puzzle with backtracking.
   - The solved puzzle Outputs `combined_result.jpg` showing the original and solved board.
3. The Flask app will display the solved image back to the user.

-----------------------------------------

The project is structured like this:

SudokuProjectFinal/
├── sudoku_ocr           ← Compiled C++ binary (place here)
├── digit_classifier.onnx
├── app.py           ← Flask app
├── uploads/         ← Uploaded images
├── templates/
│   └── index.html   ← Web UI
├──

-----------------------------------------
In order to run the sofware
1. Build the C++ Solver - Compile the C++ code using OpenCV. Just type "make all" in your terminal while in the project directory.

2. Install Flask - Make sure Flask is installed: pip install flask should do thet trick

3. Run the App
From the main directory just type "pythong3 app,py" to run the app
Then you should be able to visit http://127.0.0.1:8080 or in your browser.

-----------------------------------------

You should see
-A clean interface to upload Sudoku images
-After hitting "Solve", you’ll get redirected back the original image with the solved digits overlayed

--------------------------------------------

If there are issues 
- Make sure `digit_classifier.onnx` is in the same directory as `sudoku_ocr`
- The final result is saved as `combined_result.jpg`
- You should be able to  your own ONNX model if you'd like to improve digit recognition accuracy
--------------------------------------------
In the future I want to work on 

- Drag-and-drop UI so that it's more efficient and modern
- Support for mobile photos and skew correction to be more compatible
- Display confidence heatmaps or validation overlays

--------------------------------------------