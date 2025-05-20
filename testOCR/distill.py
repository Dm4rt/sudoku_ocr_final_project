import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define teacher model architecture (a reliable CNN for MNIST)
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)  # Changed from Dropout2d to regular Dropout
        self.dropout2 = nn.Dropout(0.5)   # Changed from Dropout2d to regular Dropout
        
        # Calculate the correct flattened size
        # After 2 pooling layers (each divides dimensions by 2), 28x28 becomes 7x7
        # With 64 channels in the last conv layer, this gives 64*7*7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fixed from 576 to 3136
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define your existing model architecture
class YourDigitModel(nn.Module):
    def __init__(self):
        super(YourDigitModel, self).__init__()
        # Features layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # features.0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # features.3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier layers
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

# Knowledge distillation trainer
class DistillationTrainer:
    def __init__(self, student_model, teacher_model, device='cpu'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device
        self.best_accuracy = 0.0
        
    def train_teacher(self, train_loader, test_loader, epochs=5):
        """Train the teacher model first"""
        print("\n== Training Teacher Model ==")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.teacher.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            # Training
            self.teacher.train()
            train_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Teacher Epoch {epoch+1}/{epochs}')
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.teacher(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': train_loss/len(progress_bar), 
                    'acc': 100.*correct/total
                })
            
            # Testing
            self.teacher.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.teacher(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            print(f'Teacher Epoch {epoch+1}: Test Loss: {test_loss/len(test_loader):.3f}, '
                  f'Test Accuracy: {100.*correct/total:.3f}%')
        
        torch.save(self.teacher.state_dict(), 'teacher_model.pth')
        print("Teacher model saved as 'teacher_model.pth'")
        return self.teacher
        
    def train(self, train_loader, test_loader, epochs=10, 
              temperature=2.0, alpha=0.5, lr=0.001):
        """
        Train student model with knowledge distillation
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            epochs: Number of training epochs
            temperature: Temperature parameter for softening distributions
            alpha: Weight for distillation loss vs. standard loss
            lr: Learning rate
        """
        print("\n== Training Student Model with Distillation ==")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        
        history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
        
        for epoch in range(epochs):
            # Training
            self.student.train()
            self.teacher.eval()  # Teacher always in eval mode
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass with student model
                student_outputs = self.student(images)
                
                # Forward pass with teacher model
                with torch.no_grad():
                    teacher_outputs = self.teacher(images)
                
                # Compute soft targets from teacher
                soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
                
                # Hard loss (cross-entropy with true labels)
                hard_loss = criterion(student_outputs, labels)
                
                # Soft loss (KL-divergence with teacher outputs)
                soft_loss = F.kl_div(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    soft_targets,
                    reduction='batchmean'
                ) * (temperature * temperature)
                
                # Combined loss
                loss = (1 - alpha) * hard_loss + alpha * soft_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Testing
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Update learning rate
            scheduler.step(test_loss)
            
            # Save best model
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                torch.save(self.student.state_dict(), 'best_distilled_model.pth')
                
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Plot training history
        self.plot_history(history)
        
        # Load best model
        self.student.load_state_dict(torch.load('best_distilled_model.pth'))
        return self.student
    
    def evaluate(self, test_loader, criterion):
        self.student.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                test_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
        
        avg_loss = test_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)
        
        return avg_loss, accuracy
    
    def evaluate_confusion_pairs(self, test_loader, pairs=[(2, 5), (8, 3)]):
        """Evaluate model performance specifically on confusing digit pairs"""
        self.student.eval()
        pair_stats = {}
        
        for digit1, digit2 in pairs:
            pair_stats[f'{digit1}_{digit2}'] = {
                f'{digit1}_correct': 0,
                f'{digit1}_total': 0,
                f'{digit2}_correct': 0,
                f'{digit2}_total': 0
            }
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                preds = outputs.argmax(dim=1)
                
                for i, (label, pred) in enumerate(zip(labels, preds)):
                    label = label.item()
                    pred = pred.item()
                    
                    for digit1, digit2 in pairs:
                        if label == digit1:
                            pair_stats[f'{digit1}_{digit2}'][f'{digit1}_total'] += 1
                            if pred == digit1:
                                pair_stats[f'{digit1}_{digit2}'][f'{digit1}_correct'] += 1
                        
                        if label == digit2:
                            pair_stats[f'{digit1}_{digit2}'][f'{digit2}_total'] += 1
                            if pred == digit2:
                                pair_stats[f'{digit1}_{digit2}'][f'{digit2}_correct'] += 1
        
        # Print results
        print("\nConfusion Pair Accuracy:")
        for pair, stats in pair_stats.items():
            digit1, digit2 = pair.split('_')
            acc1 = stats[f'{digit1}_correct'] / max(stats[f'{digit1}_total'], 1)
            acc2 = stats[f'{digit2}_correct'] / max(stats[f'{digit2}_total'], 1)
            print(f"Digit {digit1}: {acc1:.4f} ({stats[f'{digit1}_correct']}/{stats[f'{digit1}_total']})")
            print(f"Digit {digit2}: {acc2:.4f} ({stats[f'{digit2}_correct']}/{stats[f'{digit2}_total']})")
        
        return pair_stats
    
    def plot_history(self, history):
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Test Loss')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['test_acc'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Test Accuracy')
        
        plt.tight_layout()
        plt.savefig('distillation_history.png')
        plt.show()

# Main function to run the knowledge distillation process
def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create teacher model
    teacher_model = TeacherModel()
    print("Created teacher model")
    
    # Load your existing model
    your_model_path = 'sudoku_digit_model.pth'
    student_model = YourDigitModel()
    
    # Load weights from your existing model
    try:
        student_model.load_state_dict(torch.load(your_model_path))
        print("Loaded existing model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Starting with fresh model")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    # Create trainer
    trainer = DistillationTrainer(student_model, teacher_model, device)
    
    # First train the teacher if not already trained
    teacher_path = 'teacher_model.pth'
    if os.path.exists(teacher_path):
        print(f"Loading pre-trained teacher model from {teacher_path}")
        teacher_model.load_state_dict(torch.load(teacher_path))
    else:
        print("Training the teacher model first...")
        trainer.train_teacher(train_loader, test_loader, epochs=5)
    
    # Now train the student with distillation
    improved_model = trainer.train(
        train_loader, 
        test_loader,
        epochs=15,
        temperature=3.0,  # Higher temperature for softer probabilities
        alpha=0.7,        # Higher weight to teacher outputs
        lr=0.0005         # Lower learning rate for fine-tuning
    )
    
    # Evaluate specifically on confusion pairs
    trainer.evaluate_confusion_pairs(test_loader)
    
    # Save the improved model
    torch.save(improved_model.state_dict(), 'distilled_model.pth')
    print("Distilled model saved to 'distilled_model.pth'")
    
    # Convert to ONNX
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_path = 'distilled_model.onnx'
    
    torch.onnx.export(
        improved_model, 
        dummy_input, 
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to '{onnx_path}'")
    
    # Visual verification with sample digits
    visualize_predictions(improved_model, test_loader, device)

def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    
    # Get examples of previously confusing digits
    samples = {2: [], 5: [], 8: [], 3: []}
    
    with torch.no_grad():
        for images, labels in test_loader:
            for i, label in enumerate(labels):
                label = label.item()
                if label in samples and len(samples[label]) < num_samples:
                    samples[label].append(images[i])
                    
            # Check if we have enough samples
            if all(len(samples[key]) == num_samples for key in samples):
                break
    
    # Plot predictions
    plt.figure(figsize=(16, 12))
    
    for i, digit in enumerate(samples.keys()):
        for j, img in enumerate(samples[digit]):
            plt.subplot(4, num_samples, i * num_samples + j + 1)
            
            # Show image
            plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
            
            # Run prediction
            img = img.unsqueeze(0).to(device)
            output = model(img)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            pred = output.argmax(dim=1).item()
            
            # Set title with prediction
            plt.title(f"True: {digit}, Pred: {pred}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

if __name__ == "__main__":
    main()