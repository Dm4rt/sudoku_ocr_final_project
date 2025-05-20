import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# First, let's inspect the state dict to get the exact model structure
def inspect_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print("\nModel state_dict keys:")
    for key in state_dict.keys():
        print(f"  {key}: {state_dict[key].shape}")
    return state_dict

# Define the model architecture based on the state_dict
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # Features layers (convolutional part)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # features.0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # features.3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier layers (fully connected part)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),  # classifier.0
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # classifier.3
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def preprocess_image(image_path, input_size=(28, 28)):
    # Read image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to make the digit more distinct
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours found, use the largest one
    if contours:
        # Find largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Add padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2 * padding)
        h = min(thresh.shape[0] - y, h + 2 * padding)
        
        # Crop to bounding box
        thresh = thresh[y:y+h, x:x+w]
    
    # Resize to model input size
    processed = cv2.resize(thresh, input_size, interpolation=cv2.INTER_AREA)
    
    # Convert to PyTorch tensor format [batch, channels, height, width]
    tensor = torch.from_numpy(processed).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return tensor, processed

def predict_digit(model, image_path, input_size=(28, 28)):
    # Preprocess image
    tensor, processed_image = preprocess_image(image_path, input_size)
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Forward pass
        outputs = model(tensor)
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction
        confidence_scores = probs[0].cpu().numpy()
        predicted_digit = np.argmax(confidence_scores)
        
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    
    # Processed image
    plt.subplot(1, 3, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title("Processed Image")
    
    # Confidence scores
    plt.subplot(1, 3, 3)
    bars = plt.bar(range(10), confidence_scores)
    bars[predicted_digit].set_color('green')
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Confidence")
    plt.title(f"Predicted: {predicted_digit}")
    
    plt.tight_layout()
    plt.show()
    
    return predicted_digit, confidence_scores

def main():
    if len(sys.argv) < 3:
        print("Usage: python ocr_test.py <model_path> <image_path>")
        return
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Load model
    print(f"Loading PyTorch model: {model_path}")
    
    # First inspect the model to see its structure
    state_dict = inspect_model(model_path)
    
    # Create model with the appropriate architecture
    model = DigitRecognizer()
    
    # Load the state dict into the model
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded model state dictionary")
    except Exception as e:
        print(f"Error loading state dictionary: {str(e)}")
        return
    
    # Check if image path is a directory
    if os.path.isdir(image_path):
        print(f"Processing directory: {image_path}")
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                file_path = os.path.join(image_path, filename)
                print(f"\nProcessing image: {filename}")
                digit, scores = predict_digit(model, file_path)
                print(f"Predicted digit: {digit}")
                print(f"Confidence scores: {scores}")
    else:
        print(f"Processing single image: {image_path}")
        digit, scores = predict_digit(model, image_path)
        print(f"Predicted digit: {digit}")
        print(f"Confidence scores: {scores}")

if __name__ == "__main__":
    main()