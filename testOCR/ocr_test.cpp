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

namespace fs = std::filesystem;

// Function to preprocess the image for digit recognition
cv::Mat preprocessDigitImage(const cv::Mat& image, const cv::Size& inputSize) {
    cv::Mat processedImage;
    
    // Convert to grayscale if needed
    if (image.channels() > 1) {
        cv::cvtColor(image, processedImage, cv::COLOR_BGR2GRAY);
    } else {
        processedImage = image.clone();
    }
    
    // Apply thresholding to make the digit more distinct
    cv::threshold(processedImage, processedImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    
    // Find contours to isolate the digit
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
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
        
        // Crop the image to the bounding box
        processedImage = processedImage(boundingBox);
    }
    
    // Resize to the input dimensions expected by the model
    cv::resize(processedImage, processedImage, inputSize, 0, 0, cv::INTER_AREA);
    
    // Normalize pixel values to range [0, 1]
    processedImage.convertTo(processedImage, CV_32F, 1.0/255.0);
    
    return processedImage;
}

// Function to run inference with the ONNX model using OpenCV DNN
int runOCRInference(cv::dnn::Net& net, const cv::Mat& processedImage, std::vector<float>& confidenceScores) {
    // Create a blob from the processed image
    cv::Mat blob;
    if (processedImage.channels() == 1) {
        // For grayscale image, add channel dimension
        blob = cv::dnn::blobFromImage(processedImage, 1.0, cv::Size(), cv::Scalar(), false, false);
    } else {
        // For color image
        blob = cv::dnn::blobFromImage(processedImage, 1.0, cv::Size(), cv::Scalar(), false, false);
    }
    
    // Set the input blob
    net.setInput(blob);
    
    // Forward pass
    cv::Mat output = net.forward();
    
    // Process the output
    int predictedDigit = -1;
    float maxProb = -std::numeric_limits<float>::infinity();
    
    // Assuming output is a 1D array of 10 values (for digits 0-9)
    // Limit to 10 digits for display purposes
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
}

// Function to process a single image
void processImage(cv::dnn::Net& net, const cv::Size& inputSize,
                 const std::string& imagePath,
                 std::ofstream& resultsFile,
                 bool isDirectory) {
    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not read image " << imagePath << std::endl;
        return;
    }
    
    // Save original image for display
    cv::Mat originalImage = image.clone();
    
    // Preprocess the image
    cv::Mat processedImage = preprocessDigitImage(image, inputSize);
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run inference
    std::vector<float> confidenceScores;
    int predictedDigit = runOCRInference(net, processedImage, confidenceScores);
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // Display result
    std::cout << "Image: " << fs::path(imagePath).filename().string() 
              << ", Predicted digit: " << predictedDigit 
              << ", Inference time: " << duration << " ms" << std::endl;
    
    // Write to results file if open
    if (resultsFile.is_open()) {
        resultsFile << fs::path(imagePath).filename().string() << "," << predictedDigit;
        for (float score : confidenceScores) {
            resultsFile << "," << std::fixed << std::setprecision(6) << score;
        }
        resultsFile << "," << duration << std::endl;
    }
    
    // Create a results visualization
    cv::Mat resultImage;
    
    // If it's a directory mode, keep the original size for batch view
    if (isDirectory) {
        cv::resize(originalImage, resultImage, cv::Size(150, 150), 0, 0, cv::INTER_AREA);
        
        // Add the prediction label to the image
        cv::putText(resultImage, std::to_string(predictedDigit), 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                   cv::Scalar(0, 255, 0), 2);
        
        // Display this image
        std::string windowName = "Digit " + std::to_string(predictedDigit) + ": " + 
                              fs::path(imagePath).filename().string();
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, resultImage);
        cv::waitKey(100); // Brief pause to update display
    } else {
        // For single image mode, show a larger image with more details
        resultImage = cv::Mat(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));
        
        // Add original image
        cv::Mat originalResized;
        cv::resize(originalImage, originalResized, cv::Size(300, 300), 0, 0, cv::INTER_AREA);
        originalResized.copyTo(resultImage(cv::Rect(50, 50, originalResized.cols, originalResized.rows)));
        cv::putText(resultImage, "Original Image", cv::Point(50, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        // Add processed image - FIX: Convert to BGR before copying
        cv::Mat processedForDisplay;
        processedImage.convertTo(processedForDisplay, CV_8U, 255.0);
        cv::resize(processedForDisplay, processedForDisplay, cv::Size(300, 300), 0, 0, cv::INTER_AREA);
        cv::cvtColor(processedForDisplay, processedForDisplay, cv::COLOR_GRAY2BGR);
        processedForDisplay.copyTo(resultImage(cv::Rect(450, 50, processedForDisplay.cols, processedForDisplay.rows)));
        cv::putText(resultImage, "Processed Image", cv::Point(450, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        // Draw confidence bars
        int barMaxWidth = 300;
        int barHeight = 20;
        int barSpacing = 10;
        int startX = 250;
        int startY = 400;
        
        cv::putText(resultImage, "Confidence Scores", cv::Point(startX, startY - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        for (int i = 0; i < confidenceScores.size(); i++) {
            // Draw label
            cv::putText(resultImage, "Digit " + std::to_string(i) + ":", 
                       cv::Point(startX - 100, startY + i * (barHeight + barSpacing) + barHeight/2), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            
            // Draw empty bar background
            cv::rectangle(resultImage, 
                         cv::Point(startX, startY + i * (barHeight + barSpacing)), 
                         cv::Point(startX + barMaxWidth, startY + i * (barHeight + barSpacing) + barHeight), 
                         cv::Scalar(50, 50, 50), -1);
            
            // Draw filled bar based on confidence
            int barWidth = static_cast<int>(confidenceScores[i] * barMaxWidth);
            cv::Scalar barColor;
            if (i == predictedDigit) {
                barColor = cv::Scalar(0, 255, 0); // Green for the predicted digit
            } else {
                barColor = cv::Scalar(0, 128, 255); // Orange for other digits
            }
            
            cv::rectangle(resultImage, 
                         cv::Point(startX, startY + i * (barHeight + barSpacing)), 
                         cv::Point(startX + barWidth, startY + i * (barHeight + barSpacing) + barHeight), 
                         barColor, -1);
            
            // Draw confidence value
            cv::putText(resultImage, std::to_string(confidenceScores[i]).substr(0, 6), 
                       cv::Point(startX + barMaxWidth + 10, startY + i * (barHeight + barSpacing) + barHeight/2), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        
        // Add prediction and timing information
        std::string predictionText = "Predicted: " + std::to_string(predictedDigit);
        std::string timingText = "Inference time: " + std::to_string(duration) + " ms";
        
        cv::putText(resultImage, predictionText, cv::Point(50, 380), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
        cv::putText(resultImage, timingText, cv::Point(50, 420), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
        
        // Show the result
        cv::namedWindow("OCR Result", cv::WINDOW_NORMAL);
        cv::imshow("OCR Result", resultImage);
        cv::waitKey(0); // Wait for key press
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_image_or_directory> [results_file]" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string resultsFilePath = (argc > 3) ? argv[3] : "";
    
    // Check if the ONNX model file exists
    if (!fs::exists(modelPath)) {
        std::cerr << "Error: ONNX model file not found: " << modelPath << std::endl;
        return 1;
    }
    
    // Check if the image path exists
    if (!fs::exists(imagePath)) {
        std::cerr << "Error: Image path not found: " << imagePath << std::endl;
        return 1;
    }
    
    // Open results file if specified
    std::ofstream resultsFile;
    if (!resultsFilePath.empty()) {
        resultsFile.open(resultsFilePath);
        if (!resultsFile.is_open()) {
            std::cerr << "Error: Could not open results file " << resultsFilePath << std::endl;
            return 1;
        }
        
        // Write header
        resultsFile << "Filename,PredictedDigit,Confidence0,Confidence1,Confidence2,Confidence3,"
                    << "Confidence4,Confidence5,Confidence6,Confidence7,Confidence8,Confidence9,"
                    << "InferenceTime_ms" << std::endl;
    }
    
    try {
        // Load ONNX model with OpenCV DNN
        std::cout << "Loading ONNX model: " << modelPath << std::endl;
        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
        
        // Get model input size (assuming 28x28 for MNIST-like models)
        cv::Size inputSize(28, 28);
        
        // Check if model loaded successfully
        if (net.empty()) {
            std::cerr << "Error: Failed to load the ONNX model" << std::endl;
            return 1;
        }
        
        bool isDirectory = fs::is_directory(imagePath);
        
        // Check if imagePath is a directory
        if (isDirectory) {
            std::cout << "Processing directory: " << imagePath << std::endl;
            // Process all image files in the directory
            for (const auto& entry : fs::directory_iterator(imagePath)) {
                if (entry.is_regular_file()) {
                    std::string extension = entry.path().extension().string();
                    // Convert to lowercase
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    
                    // Check if it's an image file
                    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || 
                        extension == ".bmp" || extension == ".tiff" || extension == ".tif") {
                        processImage(net, inputSize, entry.path().string(), resultsFile, true);
                    }
                }
            }
        } else {
            std::cout << "Processing single image: " << imagePath << std::endl;
            // Process a single image
            processImage(net, inputSize, imagePath, resultsFile, false);
        }
        
        // Wait for key press if results are displayed
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}