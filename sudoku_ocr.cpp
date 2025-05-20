// sudoku_ocr.cpp - With improved conflict resolution

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <unordered_set>

namespace fs = std::filesystem;

// Debug flag - set to true for detailed debugging output
const bool DEBUG_MODE = true;

// Structure to store digit detection with confidence
struct DigitDetection {
    int row;
    int col;
    int digit;
    float confidence;
    
    // Constructor
    DigitDetection(int r, int c, int d, float conf) : row(r), col(c), digit(d), confidence(conf) {}
    
    // Comparison operator for sorting by confidence (highest first)
    bool operator<(const DigitDetection& other) const {
        return confidence > other.confidence; // Note: Reversed for descending order
    }
};

// Function to create a visualization of the confidence scores
cv::Mat visualizeConfidences(const std::vector<float>& confidenceScores, int predictedDigit) {
    int width = 300;
    int height = 200;
    int barMaxWidth = 250;
    int barHeight = 15;
    int barSpacing = 5;
    int startX = 40;
    int startY = 20;
    
    // Create the visualization image
    cv::Mat visualization = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Find the maximum confidence for scaling
    float maxConf = 0.0f;
    for (size_t i = 0; i < confidenceScores.size(); i++) {
        if (confidenceScores[i] > maxConf) {
            maxConf = confidenceScores[i];
        }
    }
    
    float scaleFactor = maxConf > 0 ? barMaxWidth / maxConf : 0;
    
    // Draw bars for each digit (ensure we don't exceed array bounds)
    for (size_t i = 0; i < confidenceScores.size() && i < 10; i++) {
        // Calculate y-position for this bar
        int yPos = startY + i * (barHeight + barSpacing);
        
        // Ensure we're within the image bounds
        if (yPos + barHeight > height) {
            break;
        }
        
        // Draw label
        cv::putText(visualization, std::to_string(i), 
                   cv::Point(startX - 25, yPos + barHeight/2 + 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        
        // Draw bar outline
        cv::rectangle(visualization, 
                     cv::Point(startX, yPos), 
                     cv::Point(startX + barMaxWidth, yPos + barHeight), 
                     cv::Scalar(200, 200, 200), 1);
        
        // Draw filled bar based on confidence
        int barWidth = static_cast<int>(confidenceScores[i] * scaleFactor);
        barWidth = std::min(barWidth, barMaxWidth); // Ensure we don't exceed max width
        
        cv::Scalar barColor;
        if ((int)i == predictedDigit) {
            barColor = cv::Scalar(0, 200, 0); // Green for the predicted digit
        } else {
            barColor = cv::Scalar(0, 0, 200); // Red for other digits
        }
        
        if (barWidth > 0) {
            cv::rectangle(visualization, 
                         cv::Point(startX, yPos), 
                         cv::Point(startX + barWidth, yPos + barHeight), 
                         barColor, -1);
        }
        
        // Draw confidence value
        std::string confStr = std::to_string(confidenceScores[i]);
        if (confStr.size() > 5) {
            confStr = confStr.substr(0, 5);
        }
        
        cv::putText(visualization, confStr, 
                   cv::Point(startX + barMaxWidth + 5, yPos + barHeight/2 + 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    // Add title
    cv::putText(visualization, "Digit Confidence Scores", 
               cv::Point(width/2 - 80, 10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    return visualization;
}

// Check if the center of a cell is empty (uniform color)
bool isCenterEmpty(const cv::Mat& cell) {
    try {
        if (cell.empty()) return true;
        
        // Convert to grayscale if needed
        cv::Mat grayscale;
        if (cell.channels() > 1) {
            cv::cvtColor(cell, grayscale, cv::COLOR_BGR2GRAY);
        } else {
            grayscale = cell.clone();
        }
        
        // Extract the center region (25% of width and height)
        int centerSize = std::min(grayscale.cols, grayscale.rows) / 4;
        int centerX = grayscale.cols / 2 - centerSize / 2;
        int centerY = grayscale.rows / 2 - centerSize / 2;
        
        cv::Rect centerRect(centerX, centerY, centerSize, centerSize);
        cv::Mat centerRegion = grayscale(centerRect);
        
        // Calculate mean and standard deviation of center pixels
        cv::Scalar mean, stddev;
        cv::meanStdDev(centerRegion, mean, stddev);
        
        // If standard deviation is very small, the center is uniform (empty)
        double stdDevRatio = (mean[0] > 0) ? stddev[0] / mean[0] : 0;
        
        // Debug output
        if (DEBUG_MODE) {
            std::cout << "Center region - Mean: " << mean[0] 
                     << ", StdDev: " << stddev[0] 
                     << ", Ratio: " << stdDevRatio << std::endl;
        }
        
        // Save the center region for debugging if needed
        if (DEBUG_MODE) {
            static int centerDebugCount = 0;
            cv::imwrite("debug/centers/center_" + std::to_string(centerDebugCount++) + 
                      "_stddev_" + std::to_string(stddev[0]).substr(0, 5) + ".png", centerRegion);
        }
        
        // Return true if the center is uniform (likely empty)
        return stdDevRatio < 0.05; // Adjust threshold as needed
        
    } catch (const std::exception& e) {
        std::cerr << "Error in isCenterEmpty: " << e.what() << std::endl;
        return false; // Assume not empty on error to be safe
    }
}

// Function to preprocess the image for digit recognition - Using the original OCR approach
cv::Mat preprocessDigitImage(const cv::Mat& image, const cv::Size& inputSize, bool& isEmpty) {
    try {
        if (image.empty() || image.rows == 0 || image.cols == 0) {
            isEmpty = true;
            return cv::Mat::zeros(inputSize, CV_32F);
        }
        
        // Initialize static counter for debug images
        static int debugCounter = 0;
        
        cv::Mat processedImage;
        
        // Convert to grayscale if needed
        if (image.channels() > 1) {
            cv::cvtColor(image, processedImage, cv::COLOR_BGR2GRAY);
        } else {
            processedImage = image.clone();
        }
        
        // Save original grayscale for debugging
        cv::Mat originalGray = processedImage.clone();
        
        // First, check if the center of the cell is empty
        isEmpty = isCenterEmpty(processedImage);
        
        if (isEmpty) {
            std::cout << "Cell detected as empty (center check)" << std::endl;
            
            if (DEBUG_MODE) {
                std::string baseName = "debug/processed/cell_" + std::to_string(debugCounter++);
                cv::imwrite(baseName + "_empty.png", originalGray);
            }
            
            return cv::Mat::zeros(inputSize, CV_32F);
        }
        
        // IMPORTANT: Using the original OCR code's approach for digit detection
        
        // Apply thresholding to make the digit more distinct
        cv::threshold(processedImage, processedImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        
        // Save after thresholding for debugging
        cv::Mat thresholdedImage = processedImage.clone();
        
        // Find contours to isolate the digit
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(processedImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // If contours are found, use the largest one as the digit
        if (!contours.empty()) {
            // Find the largest contour by area
            auto maxContour = std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                    return cv::contourArea(c1) < cv::contourArea(c2);
                });
            
            // Get bounding box of the largest contour
            cv::Rect boundingBox = cv::boundingRect(*maxContour);
            
            // Add a small padding
            int padding = 5;
            boundingBox.x = std::max(0, boundingBox.x - padding);
            boundingBox.y = std::max(0, boundingBox.y - padding);
            boundingBox.width = std::min(processedImage.cols - boundingBox.x, boundingBox.width + 2 * padding);
            boundingBox.height = std::min(processedImage.rows - boundingBox.y, boundingBox.height + 2 * padding);
            
            // Crop the image to the bounding box if it's valid
            if (boundingBox.width > 0 && boundingBox.height > 0 &&
                boundingBox.x + boundingBox.width <= processedImage.cols &&
                boundingBox.y + boundingBox.height <= processedImage.rows) {
                processedImage = processedImage(boundingBox).clone();
            }
        }
        
        if (DEBUG_MODE) {
            std::string baseName = "debug/processed/cell_" + std::to_string(debugCounter++);
            cv::imwrite(baseName + "_1_original.png", originalGray);
            cv::imwrite(baseName + "_2_thresholded.png", thresholdedImage);
            cv::imwrite(baseName + "_3_cropped.png", processedImage);
        }
        
        // Resize to the input dimensions expected by the model
        cv::resize(processedImage, processedImage, inputSize, 0, 0, cv::INTER_AREA);
        
        // Normalize pixel values to range [0, 1]
        processedImage.convertTo(processedImage, CV_32F, 1.0/255.0);
        
        return processedImage;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in preprocessDigitImage: " << e.what() << std::endl;
        isEmpty = true;
        return cv::Mat::zeros(inputSize, CV_32F);
    }
}

// Function to run inference with the ONNX model
int runOCRInference(cv::dnn::Net& net, const cv::Mat& processedImage, std::vector<float>& confidenceScores) {
    if (processedImage.empty() || processedImage.rows == 0 || processedImage.cols == 0) {
        std::cerr << "Warning: Empty image passed to inference" << std::endl;
        confidenceScores.resize(10, 0.0f);
        return 0;
    }
    
    // Create a blob from the processed image
    cv::Mat blob;
    try {
        if (processedImage.channels() == 1) {
            blob = cv::dnn::blobFromImage(processedImage, 1.0, cv::Size(), cv::Scalar(), false, false);
        } else {
            blob = cv::dnn::blobFromImage(processedImage, 1.0, cv::Size(), cv::Scalar(), false, false);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating blob: " << e.what() << std::endl;
        confidenceScores.resize(10, 0.0f);
        return 0;
    }
    
    // Set input and run forward pass
    try {
        net.setInput(blob);
        cv::Mat output = net.forward();
        
        // Process the output
        int predictedDigit = -1;
        float maxProb = -std::numeric_limits<float>::infinity();
        
        // Get the number of output classes (usually 10 for digits 0-9)
        int numOutputs = std::min(10, output.cols);
        
        // Store confidence scores and find the highest scoring class
        confidenceScores.resize(numOutputs);
        for (int i = 0; i < numOutputs; i++) {
            confidenceScores[i] = output.at<float>(0, i);
            if (confidenceScores[i] > maxProb) {
                maxProb = confidenceScores[i];
                predictedDigit = i;
            }
        }
        
        return predictedDigit;
    } catch (const cv::Exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        confidenceScores.resize(10, 0.0f);
        return 0;
    }
}

// Check if a number exists in the given row
bool existsInRow(const std::vector<std::vector<int>>& board, int row, int num) {
    for (int col = 0; col < 9; col++) {
        if (board[row][col] == num) {
            return true;
        }
    }
    return false;
}

// Check if a number exists in the given column
bool existsInCol(const std::vector<std::vector<int>>& board, int col, int num) {
    for (int row = 0; row < 9; row++) {
        if (board[row][col] == num) {
            return true;
        }
    }
    return false;
}

// Check if a number exists in the 3x3 box
bool existsInBox(const std::vector<std::vector<int>>& board, int boxStartRow, int boxStartCol, int num) {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            if (board[row + boxStartRow][col + boxStartCol] == num) {
                return true;
            }
        }
    }
    return false;
}

// Check if placing a digit is valid in the sudoku grid
bool isValid(const std::vector<std::vector<int>>& board, int row, int col, int num) {
    // Check row
    for (int j = 0; j < 9; j++) {
        if (board[row][j] == num) return false;
    }
    
    // Check column
    for (int i = 0; i < 9; i++) {
        if (board[i][col] == num) return false;
    }
    
    // Check 3x3 box
    int boxRow = row - row % 3;
    int boxCol = col - col % 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[boxRow + i][boxCol + j] == num) return false;
        }
    }
    
    return true;
}

// New: Check if placing a digit would be valid in the current board state
bool isValidPlacement(const std::vector<std::vector<int>>& board, int row, int col, int digit) {
    // Return false if the cell is already filled
    if (board[row][col] != 0) {
        return false;
    }
    
    // Check row
    if (existsInRow(board, row, digit)) {
        return false;
    }
    
    // Check column
    if (existsInCol(board, col, digit)) {
        return false;
    }
    
    // Check 3x3 box
    int boxStartRow = row - row % 3;
    int boxStartCol = col - col % 3;
    if (existsInBox(board, boxStartRow, boxStartCol, digit)) {
        return false;
    }
    
    return true;
}

// Check if the sudoku board is valid 
bool isValidSudokuBoard(const std::vector<std::vector<int>>& board) {
    // Check rows
    for (int i = 0; i < 9; i++) {
        std::unordered_set<int> seen;
        for (int j = 0; j < 9; j++) {
            int num = board[i][j];
            if (num != 0) {
                if (seen.find(num) != seen.end()) {
                    return false; // Duplicate in row
                }
                seen.insert(num);
            }
        }
    }
    
    // Check columns
    for (int j = 0; j < 9; j++) {
        std::unordered_set<int> seen;
        for (int i = 0; i < 9; i++) {
            int num = board[i][j];
            if (num != 0) {
                if (seen.find(num) != seen.end()) {
                    return false; // Duplicate in column
                }
                seen.insert(num);
            }
        }
    }
    
    // Check 3x3 boxes
    for (int boxRow = 0; boxRow < 3; boxRow++) {
        for (int boxCol = 0; boxCol < 3; boxCol++) {
            std::unordered_set<int> seen;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    int row = boxRow * 3 + i;
                    int col = boxCol * 3 + j;
                    int num = board[row][col];
                    if (num != 0) {
                        if (seen.find(num) != seen.end()) {
                            return false; // Duplicate in box
                        }
                        seen.insert(num);
                    }
                }
            }
        }
    }
    
    return true;
}

// NEW: Function to resolve conflicts among detected digits
void resolveConflicts(std::vector<std::vector<int>>& board, 
                     std::vector<std::vector<float>>& confidenceMap,
                     std::vector<DigitDetection>& tentativeDigits) {
    
    if (tentativeDigits.empty()) {
        return; // No tentative digits to process
    }
    
    std::cout << "\nResolving conflicts among " << tentativeDigits.size() 
              << " tentative digits...\n";
    
    // Sort tentative digits by confidence (highest first)
    std::sort(tentativeDigits.begin(), tentativeDigits.end());
    
    // First pass: Try to add tentative digits if they don't violate Sudoku rules
    int digitsAdded = 0;
    for (const auto& detection : tentativeDigits) {
        int row = detection.row;
        int col = detection.col;
        int digit = detection.digit;
        
        // Skip if this cell already has a high-confidence digit
        if (board[row][col] != 0) {
            continue;
        }
        
        // Check if placing this digit would be valid
        if (isValidPlacement(board, row, col, digit)) {
            // Place the digit
            board[row][col] = digit;
            confidenceMap[row][col] = detection.confidence;
            digitsAdded++;
            
            std::cout << "Added tentative digit " << digit 
                      << " at [" << row << "][" << col << "] with confidence " 
                      << detection.confidence << std::endl;
        } else {
            std::cout << "Rejected tentative digit " << digit 
                      << " at [" << row << "][" << col << "] - would violate Sudoku rules" 
                      << std::endl;
        }
    }
    
    std::cout << "Added " << digitsAdded << " tentative digits after conflict resolution." << std::endl;
    
    // Check if the board is now valid
    if (!isValidSudokuBoard(board)) {
        std::cout << "Warning: Board still invalid after adding tentative digits." << std::endl;
        
        // Find and fix remaining conflicts
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                // Skip empty cells
                if (board[i][j] == 0) continue;
                
                // Make a copy of the board without this digit
                std::vector<std::vector<int>> testBoard = board;
                int currentDigit = testBoard[i][j];
                testBoard[i][j] = 0;
                
                // If removing this digit makes the board valid, it's part of a conflict
                if (!isValidSudokuBoard(board) && isValidSudokuBoard(testBoard)) {
                    std::cout << "Removing conflicting digit " << currentDigit 
                              << " at [" << i << "][" << j << "] with confidence " 
                              << confidenceMap[i][j] << std::endl;
                    
                    // Remove this digit from the actual board
                    board[i][j] = 0;
                    confidenceMap[i][j] = 0.0f;
                }
            }
        }
    }
}

// Solve the sudoku puzzle using backtracking
bool solveSudoku(std::vector<std::vector<int>>& board) {
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            // Find an empty cell
            if (board[row][col] == 0) {
                // Try digits 1-9
                for (int num = 1; num <= 9; num++) {
                    if (isValid(board, row, col, num)) {
                        // Place the digit if it's valid
                        board[row][col] = num;
                        
                        // Recursively solve the rest of the board
                        if (solveSudoku(board)) {
                            return true;
                        }
                        
                        // If placing the digit doesn't lead to a solution, backtrack
                        board[row][col] = 0;
                    }
                }
                // No digit works in this cell
                return false;
            }
        }
    }
    // All cells filled
    return true;
}

// Create directory if it doesn't exist
void createDirectory(const std::string& path) {
    try {
        if (!fs::exists(path)) {
            fs::create_directories(path);
            std::cout << "Created directory: " << path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
}

// Draw grid lines on an image
void drawSudokuGrid(cv::Mat& image, const cv::Scalar& color) {
    int height = image.rows;
    int width = image.cols;
    int cellSize = width / 9;
    
    // Draw horizontal lines
    for (int i = 0; i <= 9; i++) {
        int thickness = (i % 3 == 0) ? 2 : 1;
        cv::line(image, cv::Point(0, i * cellSize), cv::Point(width, i * cellSize), 
                color, thickness);
    }
    
    // Draw vertical lines
    for (int i = 0; i <= 9; i++) {
        int thickness = (i % 3 == 0) ? 2 : 1;
        cv::line(image, cv::Point(i * cellSize, 0), cv::Point(i * cellSize, height), 
                color, thickness);
    }
}

// Save a visualization of all cells and their detection status
void saveSimplifiedDebugGrid(const std::vector<std::vector<int>>& board, 
                            const std::vector<std::vector<float>>& confidenceMap,
                            const std::string& filename) {
    int cellSize = 50;
    int gridSize = cellSize * 9;
    cv::Mat debugGrid(gridSize, gridSize, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw grid lines
    drawSudokuGrid(debugGrid, cv::Scalar(0, 0, 0));
    
    // Add digits to cells
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            int digit = board[i][j];
            float confidence = confidenceMap[i][j];
            
            // Calculate text position
            cv::Point textPos(j * cellSize + cellSize/3, (i+1) * cellSize - cellSize/3);
            
            if (digit > 0) {
                // Color based on confidence
                cv::Scalar color;
                if (confidence > 2.0) {
                    color = cv::Scalar(0, 255, 0); // High confidence - green
                } else if (confidence > 1.0) {
                    color = cv::Scalar(0, 255, 255); // Medium confidence - yellow
                } else {
                    color = cv::Scalar(0, 0, 255); // Low confidence - red
                }
                
                // Draw digit
                cv::putText(debugGrid, std::to_string(digit), textPos, 
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
                
                // Draw confidence value (smaller)
                std::string confStr = std::to_string(confidence);
                if (confStr.size() > 4) {
                    confStr = confStr.substr(0, 4);
                }
                
                cv::putText(debugGrid, confStr, 
                           cv::Point(j * cellSize + 5, (i+1) * cellSize - 5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
            } else {
                // Empty cell - just draw a small "0"
                cv::putText(debugGrid, "0", 
                           cv::Point(j * cellSize + cellSize/3, (i+1) * cellSize - cellSize/3), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);
            }
        }
    }
    
    // Save the debug grid
    cv::imwrite(filename, debugGrid);
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cout << "Usage: ./sudoku_ocr <sudoku_image.jpg> <digit_classifier.onnx> [confidence_threshold]" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    std::string modelPath = argv[2];
    
    // Default confidence threshold
    float confidenceThreshold = 0.8;
    float highConfidenceThreshold = 1.5; // Threshold for first pass (high confidence)
    
    // Optional confidence threshold from command line
    if (argc == 4) {
        confidenceThreshold = std::stof(argv[3]);
        highConfidenceThreshold = confidenceThreshold * 1.5;
    }
    
    // Check if the ONNX model file exists
    if (!fs::exists(modelPath)) {
        std::cerr << "Error: ONNX model file not found: " << modelPath << std::endl;
        return 1;
    }

    // Create directories for output
    try {
        createDirectory("cells");
        createDirectory("debug");
        createDirectory("debug/processed");
        createDirectory("debug/confidences");
        createDirectory("debug/centers");
    } catch (const std::exception& e) {
        std::cerr << "Error creating directories: " << e.what() << std::endl;
    }
    
    // 1. Process the sudoku image
    cv::Mat image;
    try {
        image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Image not found or could not be loaded: " << imagePath << std::endl;
            return -1;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading image: " << e.what() << std::endl;
        return -1;
    }

    // Store original image for final visualization
    cv::Mat originalImage = image.clone();
    
    // Convert to grayscale
    cv::Mat gray;
    try {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } catch (const cv::Exception& e) {
        std::cerr << "Error converting to grayscale: " << e.what() << std::endl;
        return -1;
    }
    
    // Apply bilateral filter to preserve edges while reducing noise
    cv::Mat blurred;
    try {
        cv::bilateralFilter(gray, blurred, 9, 75, 75);
        cv::imwrite("debug/1_bilateral_filtered.jpg", blurred);
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying bilateral filter: " << e.what() << std::endl;
        blurred = gray.clone(); // Use original if filtering fails
    }
    
    // Apply adaptive threshold 
    cv::Mat thresh;
    try {
        cv::adaptiveThreshold(blurred, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY_INV, 11, 2);
        cv::imwrite("debug/2_thresh_adaptive.jpg", thresh);
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying threshold: " << e.what() << std::endl;
        return -1;
    }

    // Find all contours
    std::vector<std::vector<cv::Point>> contours;
    try {
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    } catch (const cv::Exception& e) {
        std::cerr << "Error finding contours: " << e.what() << std::endl;
        return -1;
    }
    
    // Find largest quadrilateral contour (likely the sudoku grid)
    double maxArea = 0;
    std::vector<cv::Point> bestContour;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        
        // Skip small contours
        if (area < image.cols * image.rows * 0.05) // Minimum 5% of image
            continue;
        
        // Approximate contour with a polygon
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);
        
        // If it's a quadrilateral and larger than current best, update best
        if (approx.size() == 4 && area > maxArea) {
            bestContour = approx;
            maxArea = area;
        }
    }

    if (bestContour.empty()) {
        std::cerr << "No Sudoku board found. Try adjusting the contour detection parameters." << std::endl;
        return -1;
    }
    
    // Draw the detected contour on the original image for debugging
    try {
        cv::Mat contourImage = originalImage.clone();
        std::vector<std::vector<cv::Point>> bestContours = {bestContour};
        cv::drawContours(contourImage, bestContours, 0, cv::Scalar(0, 255, 0), 2);
        cv::imwrite("debug/3_detected_contour.jpg", contourImage);
    } catch (const cv::Exception& e) {
        std::cerr << "Error drawing contours: " << e.what() << std::endl;
    }

    // Sort points to get top-left, top-right, bottom-right, bottom-left order
    auto sortPoints = [](std::vector<cv::Point> pts) {
        if (pts.size() != 4) return pts;
        
        // Calculate center point
        cv::Point center(0, 0);
        for (const auto& pt : pts) {
            center.x += pt.x;
            center.y += pt.y;
        }
        center.x /= 4;
        center.y /= 4;
        
        // Sort points based on their position relative to center
        std::vector<cv::Point> sorted(4);
        std::vector<cv::Point> tl, tr, bl, br;
        
        for (const auto& pt : pts) {
            if (pt.x < center.x && pt.y < center.y)
                tl.push_back(pt);
            else if (pt.x >= center.x && pt.y < center.y)
                tr.push_back(pt);
            else if (pt.x < center.x && pt.y >= center.y)
                bl.push_back(pt);
            else
                br.push_back(pt);
        }
        
        // Handle cases where points don't fall neatly into quadrants
        if (tl.empty() || tr.empty() || bl.empty() || br.empty()) {
            // Sort by y first to get top and bottom points
            std::sort(pts.begin(), pts.end(), [](cv::Point a, cv::Point b) { return a.y < b.y; });
            
            std::vector<cv::Point> top = {pts[0], pts[1]};
            std::vector<cv::Point> bottom = {pts[2], pts[3]};
            
            // Sort each pair by x
            std::sort(top.begin(), top.end(), [](cv::Point a, cv::Point b) { return a.x < b.x; });
            std::sort(bottom.begin(), bottom.end(), [](cv::Point a, cv::Point b) { return a.x < b.x; });
            
            // Return ordered points: top-left, top-right, bottom-right, bottom-left
            return std::vector<cv::Point>{top[0], top[1], bottom[1], bottom[0]};
        } else {
            // If we have points in each quadrant, just take one from each
            return std::vector<cv::Point>{tl[0], tr[0], br[0], bl[0]};
        }
    };

    try {
        bestContour = sortPoints(bestContour);
    } catch (const std::exception& e) {
        std::cerr << "Error sorting points: " << e.what() << std::endl;
        return -1;
    }
    
    // Convert to Point2f for perspective transform
    std::vector<cv::Point2f> src;
    for (const auto& pt : bestContour) {
        src.push_back(cv::Point2f(pt));
    }
    
    // Ensure we have exactly 4 points
    if (src.size() != 4) {
        std::cerr << "Error: Need exactly 4 points for perspective transform, got " << src.size() << std::endl;
        return -1;
    }
    
    // Target points for perspective transform (square grid)
    int gridSize = 450; // Size of the warped grid
    std::vector<cv::Point2f> dst = { 
        cv::Point2f(0, 0), cv::Point2f(gridSize, 0), 
        cv::Point2f(gridSize, gridSize), cv::Point2f(0, gridSize) 
    };
    
    // Compute perspective transform matrix
    cv::Mat matrix;
    try {
        matrix = cv::getPerspectiveTransform(src, dst);
    } catch (const cv::Exception& e) {
        std::cerr << "Error computing perspective transform: " << e.what() << std::endl;
        return -1;
    }

    // Warp the image to get a straight view of the sudoku
    cv::Mat warped;
    try {
        cv::warpPerspective(gray, warped, matrix, cv::Size(gridSize, gridSize));
        cv::imwrite("warped_sudoku.jpg", warped);
    } catch (const cv::Exception& e) {
        std::cerr << "Error warping perspective: " << e.what() << std::endl;
        return -1;
    }
    
    // Calculate cell size
    int cellSize = gridSize / 9;

    // 2. Load the OCR model
    std::cout << "Loading ONNX model: " << modelPath << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty()) {
            std::cerr << "Error: Failed to load the ONNX model" << std::endl;
            return 1;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        return 1;
    }
    
    cv::Size inputSize(28, 28); // Standard input size for MNIST-like models
    
    // 3. Initialize sudoku board
    std::vector<std::vector<int>> sudokuBoard(9, std::vector<int>(9, 0));
    std::vector<std::vector<float>> confidenceMap(9, std::vector<float>(9, 0.0f));
    
    // NEW: vector to store tentative digit detections for conflict resolution
    std::vector<DigitDetection> tentativeDigits;
    
    // 4. Extract and recognize digits in each cell
    std::cout << "Analyzing cells and recognizing digits...\n";
    
    // Process each cell
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            try {
                // Extract the cell region
                int x = j * cellSize;
                int y = i * cellSize;
                int width = cellSize;
                int height = cellSize;
                
                if (x + width > warped.cols || y + height > warped.rows) {
                    std::cerr << "Cell dimensions out of bounds" << std::endl;
                    continue;
                }
                
                cv::Rect cellRect(x, y, width, height);
                cv::Mat cell = warped(cellRect).clone();
                
                // Save the original cell
                std::string cellFilename = "cells/cell_" + std::to_string(i) + "_" + std::to_string(j) + ".png";
                cv::imwrite(cellFilename, cell);
                
                // Preprocess the cell for OCR with center check for empty cells
                bool isEmpty = false;
                cv::Mat processedCell = preprocessDigitImage(cell, inputSize, isEmpty);
                
                // Skip empty cells
                if (isEmpty) {
                    sudokuBoard[i][j] = 0;
                    confidenceMap[i][j] = 0.0f;
                    continue;
                }
                
                // Run inference on non-empty cells
                std::vector<float> confidenceScores;
                int digit = runOCRInference(net, processedCell, confidenceScores);
                
                // Create and save confidence visualization
                try {
                    cv::Mat confidenceViz = visualizeConfidences(confidenceScores, digit);
                    std::string confFilename = "debug/confidences/cell_" + std::to_string(i) + "_" + 
                                             std::to_string(j) + "_conf.png";
                    cv::imwrite(confFilename, confidenceViz);
                } catch (const cv::Exception& e) {
                    std::cerr << "Error creating confidence visualization: " << e.what() << std::endl;
                }
                
                // UPDATED APPROACH: Two-stage digit acceptance
                // First pass: Only accept very high confidence detections without constraint checks
                if (digit > 0 && confidenceScores[digit] >= highConfidenceThreshold) {
                    sudokuBoard[i][j] = digit;
                    confidenceMap[i][j] = confidenceScores[digit];
                    
                    std::cout << "Cell [" << i << "][" << j << "]: Detected digit " << digit 
                              << " with high confidence " << confidenceScores[digit] << std::endl;
                } 
                // Store medium confidence detections for later conflict resolution
                else if (digit > 0 && confidenceScores[digit] >= confidenceThreshold) {
                    // Store as tentative detection
                    tentativeDigits.push_back(DigitDetection(i, j, digit, confidenceScores[digit]));
                    
                    std::cout << "Cell [" << i << "][" << j << "]: Tentative digit " << digit 
                              << " with medium confidence " << confidenceScores[digit] << std::endl;
                } else {
                    // Empty or low confidence
                    sudokuBoard[i][j] = 0;
                    confidenceMap[i][j] = 0.0f;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing cell [" << i << "][" << j << "]: " << e.what() << std::endl;
                // Skip this cell on error
                sudokuBoard[i][j] = 0;
                confidenceMap[i][j] = 0.0f;
            }
        }
    }
    
    // Save a debug grid showing high confidence digits only
    try {
        saveSimplifiedDebugGrid(sudokuBoard, confidenceMap, "debug/initial_high_confidence_grid.jpg");
    } catch (const std::exception& e) {
        std::cerr << "Error saving debug grid: " << e.what() << std::endl;
    }
    
    // Now resolve conflicts among tentative digits
    resolveConflicts(sudokuBoard, confidenceMap, tentativeDigits);
    
    // Save a debug grid after conflict resolution
    try {
        saveSimplifiedDebugGrid(sudokuBoard, confidenceMap, "debug/after_conflict_resolution_grid.jpg");
    } catch (const std::exception& e) {
        std::cerr << "Error saving debug grid: " << e.what() << std::endl;
    }
    
    // 5. Validate the detected board - this is now a final sanity check
    if (!isValidSudokuBoard(sudokuBoard)) {
        std::cout << "Warning: Final detected sudoku board still contains errors." << std::endl;
        std::cout << "Attempting last-resort fixes by removing lowest confidence digits..." << std::endl;
        
        // Try removing digits one by one, starting with lowest confidence
        std::vector<std::tuple<int, int, float>> digitConfidences;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (sudokuBoard[i][j] > 0) {
                    digitConfidences.push_back(std::make_tuple(i, j, confidenceMap[i][j]));
                }
            }
        }
        
        // Sort by confidence (ascending)
        std::sort(digitConfidences.begin(), digitConfidences.end(), 
                 [](const auto& a, const auto& b) {
                     return std::get<2>(a) < std::get<2>(b);
                 });
        
        // Try removing the lowest confidence digits until the board is valid
        for (const auto& digit : digitConfidences) {
            int row = std::get<0>(digit);
            int col = std::get<1>(digit);
            float conf = std::get<2>(digit);
            
            // Temporarily remove the digit
            int originalValue = sudokuBoard[row][col];
            sudokuBoard[row][col] = 0;
            confidenceMap[row][col] = 0.0f;
            
            std::cout << "Removing low confidence digit " << originalValue 
                      << " at [" << row << "][" << col << "] with confidence " 
                      << conf << std::endl;
            
            // Check if board is now valid
            if (isValidSudokuBoard(sudokuBoard)) {
                std::cout << "Board is now valid after removing digit." << std::endl;
                break;
            }
        }
    }
    
    // 6. Print the detected board
    std::cout << "\nDetected Sudoku Board:" << std::endl;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (sudokuBoard[i][j] == 0) {
                std::cout << ". ";
            } else {
                std::cout << sudokuBoard[i][j] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // Save simplified debug grid of final board
    try {
        saveSimplifiedDebugGrid(sudokuBoard, confidenceMap, "debug/final_board.jpg");
    } catch (const std::exception& e) {
        std::cerr << "Error saving final debug grid: " << e.what() << std::endl;
    }
    
    // 7. Create a copy of the board for solving
    std::vector<std::vector<int>> solvedBoard = sudokuBoard;
    
    // 8. Solve the sudoku
    std::cout << "\nSolving the puzzle...\n";
    bool hasSolution = solveSudoku(solvedBoard);
    
    if (hasSolution) {
        std::cout << "\nSolved Sudoku Board:" << std::endl;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                std::cout << solvedBoard[i][j] << " ";
            }
            std::cout << std::endl;
        }
        
        // 9. Draw the solved board on the warped image
        cv::Mat solvedImage;
        try {
            cv::cvtColor(warped, solvedImage, cv::COLOR_GRAY2BGR);
            
            for (int i = 0; i < 9; ++i) {
                for (int j = 0; j < 9; ++j) {
                    cv::Point textPosition(j * cellSize + cellSize/4, (i+1) * cellSize - cellSize/3);
                    
                    if (sudokuBoard[i][j] == 0 && solvedBoard[i][j] != 0) {
                        // Draw filled cells in red
                        cv::putText(solvedImage, std::to_string(solvedBoard[i][j]), 
                                   textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
                    } else if (sudokuBoard[i][j] != 0) {
                        // Draw original digits in green
                        cv::putText(solvedImage, std::to_string(sudokuBoard[i][j]), 
                                   textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
            
            // Draw grid lines
            drawSudokuGrid(solvedImage, cv::Scalar(0, 255, 0));
            
            // Save the solved image
            cv::imwrite("solved_sudoku.jpg", solvedImage);
            std::cout << "Solved Sudoku saved to solved_sudoku.jpg" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error creating solved image: " << e.what() << std::endl;
        }
        
        // 10. Create a combined original vs solved image
        try {
            cv::Mat combined;
            cv::hconcat(originalImage, cv::Mat(originalImage.size(), originalImage.type(), cv::Scalar(255, 255, 255)), combined);
            
            // Create the inverse perspective transform matrix
            cv::Mat invMatrix = cv::getPerspectiveTransform(dst, src);
            
            // Transform the solved sudoku back to the original perspective
            cv::Mat solvedOriginal = originalImage.clone();
            cv::Mat solvedWarped;
            
            // Warp the solved image back to the original perspective
            cv::warpPerspective(solvedImage, solvedWarped, invMatrix, originalImage.size());
            
            // Create a mask for the sudoku region
            cv::Mat mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
            std::vector<cv::Point> polygonPoints = {bestContour[0], bestContour[1], bestContour[2], bestContour[3]};
            std::vector<std::vector<cv::Point>> polygons = {polygonPoints};
            cv::fillPoly(mask, polygons, cv::Scalar(255));
            
            // Copy the warped solved image to the original image using the mask
            solvedWarped.copyTo(solvedOriginal, mask);
            
            // Place the solved image in the second half of the combined image
            solvedOriginal.copyTo(combined(cv::Rect(originalImage.cols, 0, originalImage.cols, originalImage.rows)));
            
            // Add labels
            cv::putText(combined, "Original", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(combined, "Solved", cv::Point(originalImage.cols + 10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            
            cv::imwrite("combined_result.jpg", combined);
            std::cout << "Combined result saved to combined_result.jpg" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error creating combined image: " << e.what() << std::endl;
        }
        
    } else {
        std::cout << "No solution found for the sudoku board." << std::endl;
    }
    
    return 0;
}