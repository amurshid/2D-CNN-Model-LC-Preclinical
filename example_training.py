# ------------------------------------------------------------------------
# Example Training Script for Lung Cancer CT Scan Classification
# This script demonstrates how to train a 2D CNN model using the preprocessed data.
# ------------------------------------------------------------------------

"""
Example Training Script for Lung Cancer CT Scan Classification
This script demonstrates how to train a 2D CNN model using the preprocessed data.
"""

# -------------------- Import Required Libraries --------------------

import torch                                           # Import the main PyTorch library for deep learning
import torch.nn as nn                                  # Import PyTorch's neural network subpackage (layers, modules, etc.)
import torch.optim as optim                            # Import PyTorch's optimization package (optimizers like Adam)
from torch.utils.data import DataLoader                # Import DataLoader utility for batching/shuffling/loading datasets
from data_loader import create_data_loaders            # Custom function to create data loaders for this dataset/project
import numpy as np                                     # Import NumPy library for numerical operations and array handling
from tqdm import tqdm                                  # Progress bar library to make loops visually trackable
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (saves to file without requiring display)
import matplotlib.pyplot as plt                        # Matplotlib for plotting training curves (loss, accuracy, etc.)

# Additional imports for metrics, confusion matrix, and tabular results
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
)
import pandas as pd

# -------------------- Define the Neural Network Model --------------------

class SimpleLungCancerCNN(nn.Module):
    """
    Simple CNN model for lung cancer classification.
    """
    def __init__(self, num_classes=5):
        # Call the parent constructor to initialize the nn.Module class
        super(SimpleLungCancerCNN, self).__init__()
        
        # Define a stack of convolutional and pooling layers for feature extraction.
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),      # 2D convolution: input 3 channels, output 64 channels, 3x3 kernel, 1-pixel padding
            nn.BatchNorm2d(64),                              # Batch normalization for 64 feature maps
            nn.ReLU(inplace=True),                           # In-place ReLU non-linearity
            nn.MaxPool2d(2, 2),                              # 2x2 max pooling reduces width and height by half
            nn.Dropout2d(0.25),                              # Drop out 25% of features randomly during training
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # Input: 64 feature maps, Output: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # Input: 128 feature maps, Output: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),   # Input: 256 feature maps, Output: 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier "head": Reduce spatial features into class logits for classification.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),                   # Pool any size to a fixed (7,7) output for further processing
            nn.Flatten(),                                   # Flatten to 1D for input to Linear layer
            nn.Linear(512 * 7 * 7, 512),                    # Fully connected layer: all features to 512 units
            nn.BatchNorm1d(512),                            # Normalize activations across batch
            nn.ReLU(inplace=True),                          # Activation function
            nn.Dropout(0.5),                                # Drop out half the neurons' outputs randomly at train time
            nn.Linear(512, 256),                            # Second fully connected layer: 512 to 256 units
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                                # Drop out half again
            nn.Linear(256, num_classes)                     # Final fully connected: units → number of classes (logits)
        )
    
    def forward(self, x):
        # Defines the forward pass for input x through entire network
        x = self.features(x)   # Feature extraction by conv layers and poolings
        x = self.classifier(x) # Flatten and classify features
        return x

# -------------------- Single Epoch Training Function --------------------

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()                          # Set model into training mode (enables dropout, updates BN stats)
    running_loss = 0.0                     # Initialize variable to accumulate total epoch loss
    correct = 0                            # Tracks number of correct predictions
    total = 0                              # Tracks total samples processed

    # Iterate over every batch in the training data loader
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)   # Move input data and labels to CPU or GPU as needed
        optimizer.zero_grad()                                   # Zero all previous gradients before this step

        outputs = model(images)                                 # Forward pass: run images through model to get predictions
        loss = criterion(outputs, labels)                       # Compute loss value against the ground truth
        
        loss.backward()                                         # Backward pass: compute gradients with respect to loss
        optimizer.step()                                        # Update model weights/parameters using optimizer
        
        running_loss += loss.item()                             # Add loss value for this batch to the total
        _, predicted = torch.max(outputs.data, 1)               # Take the class with the highest score as prediction
        total += labels.size(0)                                 # Increase total counter by batch size
        correct += (predicted == labels).sum().item()           # Increase correct counter if prediction matches ground truth
    
    epoch_loss = running_loss / len(train_loader)               # Compute average loss by dividing total by number of batches
    epoch_acc = 100 * correct / total                           # Calculate accuracy as percentage (correct / total)
    return epoch_loss, epoch_acc                                # Return average loss and accuracy

# -------------------- Validation (Evaluation) Function --------------------

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    Returns:
        Average loss and accuracy
    """
    model.eval()                                            # Set model to evaluation mode (turns off dropout, BN uses running stats)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():                                   # Don't calculate gradients; saves memory and time
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)  # Send to device
            outputs = model(images)                               # Get model predictions
            loss = criterion(outputs, labels)                     # Compute loss
            
            running_loss += loss.item()                           # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)             # Predicted labels
            total += labels.size(0)                               # Count
            correct += (predicted == labels).sum().item()         # Count correct predictions
    
    epoch_loss = running_loss / len(val_loader)                   # Average loss over batches
    epoch_acc = 100 * correct / total                             # Compute percent accuracy
    return epoch_loss, epoch_acc

# -------------------- Full Training Pipeline --------------------

def train_model(data_dir="processed_data", num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Train the CNN model.
    Args:
        data_dir: Path to processed data directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    # ---- Setup device for computation ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Prefer GPU, fallback to CPU
    print(f"Using device: {device}")

    # ---- Create data loaders for train, val, and test sets ----
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir,         # The location of the processed dataset 
        batch_size=batch_size,     # Number of samples per batch
        num_workers=0,             # Data loading in main process/thread
        augment=True               # Enable data augmentation
    )
    print(f"Classes: {class_names}")                                 # Print class names
    print(f"Number of classes: {len(class_names)}")                  # Print number of distinct classes

    # ---- Instantiate the CNN model ----
    model = SimpleLungCancerCNN(num_classes=len(class_names))        # Use number of classes from data loader
    model = model.to(device)                                         # Move model to computation device
    # ---- Loss function, optimizer and learning rate scheduler ----
    criterion = nn.CrossEntropyLoss()                                # Use cross-entropy for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Adam optimizer with weight decay regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )                                                               # Scheduler reduces learning rate when validation loss plateaus

    # ---- Initialize dictionary to track loss/accuracy through training ----
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0.0                                              # Track the best (highest) validation accuracy achieved
    best_model_state = None                                         # This will store the parameters of the best model

    print("\nStarting training...")
    print("=" * 60)
    # ---- Main training loop over epochs ----
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")                    # Print which epoch we're on (1-based for humans)
        print("-" * 60)
        # Train the model for one full epoch, collecting training loss & accuracy
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate ("validate") the model on the validation dataset, collecting loss & accuracy
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        # Only reduce LR if the validation loss does not improve for several epochs
        scheduler.step(val_loss)

        # Save loss and accuracy results so we can plot them later
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print metrics for this epoch: summary of how well model did
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")        # Show current LR since it may change

        # If this epoch gave us a new best validation accuracy, save the model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc                                            # Update best accuracy tracker
            best_model_state = model.state_dict().copy()                      # Save current model weights
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
            torch.save(best_model_state, 'best_model.pth')                    # Also save to file (for safety)

    print("\n" + "=" * 60)
    print("Training completed!")                                              # End of training loop
    print(f"Best validation accuracy: {best_val_acc:.2f}%")                  # Report the highest val acc reached

    # ---- Testing phase: Evaluate the best model on the hold-out test set ----
    if best_model_state:
        model.load_state_dict(best_model_state)                               # Restore best weights
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)     # Compute test metrics
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")          # Print test results

    # ---- Plot the loss/accuracy training history ----
    plot_training_history(history)                                            # Draw and save the curves

    # ---- ADDED: Advanced Metrics, Graphs, and Truth Table after Testing ----
    print('\nCalculating per-class metrics and generating plots...')

    # 1. Obtain predictions, true labels, losses, and image indices for the test set
    all_labels = []
    all_preds = []
    all_losses = []
    all_indices = []
    test_loss_per_sample = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting Test Results")):
            # Try to access indices if available (e.g. test_loader returns (img, label, idx))
            if len(batch) == 3:
                images, labels, indices = batch
                all_indices.extend(indices.cpu().numpy())
            else:
                images, labels = batch
                indices = np.arange(len(labels)) + len(all_labels)
                all_indices.extend(indices)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_per_sample.extend([loss.item()] * len(labels))
            _, preds = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    test_loss_per_sample = np.array(test_loss_per_sample)
    all_indices = np.array(all_indices)

    # 2. Compute classification report and per-class metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    confmat = confusion_matrix(all_labels, all_preds)
    support = np.array([report[c]['support'] for c in class_names])
    precision = np.array([report[c]['precision'] for c in class_names])
    recall    = np.array([report[c]['recall'] for c in class_names])
    f1score   = np.array([report[c]['f1-score'] for c in class_names])
    accuracy  = np.array([confmat.diagonal() / support])  # Per-class accuracy

    # 3. Plot confusion matrix, per-class metrics bar charts, and per-class loss
    # (a) Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    im = ax_cm.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im, ax=ax_cm)
    tick_marks = np.arange(len(class_names))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
    ax_cm.set_yticklabels(class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = confmat.max() / 2.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax_cm.text(j, i, format(confmat[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if confmat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close(fig_cm)
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")

    # (b) Per-class precision, recall, f1-score, accuracy bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].bar(class_names, precision)
    axes[0].set_title('Per-Class Precision')
    axes[0].set_ylabel('Precision')
    axes[1].bar(class_names, recall)
    axes[1].set_title('Per-Class Recall')
    axes[1].set_ylabel('Recall')
    axes[2].bar(class_names, f1score)
    axes[2].set_title('Per-Class F1-score')
    axes[2].set_ylabel('F1-score')
    axes[3].bar(class_names, accuracy[0])
    axes[3].set_title('Per-Class Accuracy')
    axes[3].set_ylabel('Accuracy')
    for ax in axes:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('per_class_metrics.png', bbox_inches='tight')
    plt.close(fig)
    print("Per-class metrics bar chart saved as 'per_class_metrics.png'.")

    # (c) Per-class average loss plot (approximate)
    class_loss = []
    for cid in range(len(class_names)):
        indices_class = np.where(all_labels == cid)[0]
        class_losses = []
        for idx in indices_class:
            # Individual loss: if cross entropy is per batch, approximate by batch average
            # here we just take average loss for batch, so all in batch get same value
            class_losses.append(test_loss_per_sample[idx])
        class_loss.append(np.mean(class_losses) if class_losses else 0)
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, class_loss)
    plt.ylabel('Average Loss')
    plt.title('Per-Class Average Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('per_class_avg_loss.png', bbox_inches='tight')
    plt.close()
    print("Per-class average loss plot saved as 'per_class_avg_loss.png'.")

    # (d) Precision-Recall curve per class (optional if you want, more complex for multi-class)
    # Can be added if more metrics required

    # 4. Generate and save a truth table for misclassified images (images "lost")
    # If test_loader yields the path or index, record that
    lost_samples = []
    for i, (true_label, pred_label, idx) in enumerate(zip(all_labels, all_preds, all_indices)):
        if true_label != pred_label:
            lost_samples.append({'Sample_Index': idx, 
                                'True_Label': class_names[true_label], 
                                'Predicted_Label': class_names[pred_label]})
    truth_table_df = pd.DataFrame(lost_samples)
    truth_table_df.index.name = 'Lost_Image_Number'
    truth_table_df.to_csv('misclassified_truth_table.csv')
    print(f"Truth table for lost images saved to 'misclassified_truth_table.csv' ({len(lost_samples)} images lost).")

    # 5. Print brief report summary
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print(f"\nTotal test samples: {len(all_labels)}")
    print(f"Number of misclassified samples: {len(lost_samples)}")
    print("Additional evaluation plots and truth table saved to disk.")

    return model, history                                                    # Return trained model and stats

# -------------------- Plotting Function for Training History --------------------

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    """
    # Create a 1-row, 2-column subplot -- one panel for loss, one for accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Plot loss over epochs ----
    ax1.plot(history['train_loss'], label='Train Loss')                      # Training loss per epoch
    ax1.plot(history['val_loss'], label='Val Loss')                          # Validation loss per epoch
    ax1.set_xlabel('Epoch')                                                  # Label x-axis
    ax1.set_ylabel('Loss')                                                   # Label y-axis
    ax1.set_title('Training and Validation Loss')                            # Title for loss panel
    ax1.legend()                                                             # Add legend
    ax1.grid(True)                                                           # Add grid for easier visualization

    # ---- Plot accuracy over epochs ----
    ax2.plot(history['train_acc'], label='Train Acc')                        # Training accuracy per epoch
    ax2.plot(history['val_acc'], label='Val Acc')                            # Validation accuracy per epoch
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # ---- Adjust layout and save the figure ----
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')        # Save the plot as a static PNG file
    print("\nTraining history saved to 'training_history.png'")              # Tell user where to find figure
    plt.close()  # Close the figure to free memory

# -------------------- Script Entry Point --------------------

if __name__ == "__main__":
    # When run as a script (not imported as a module),
    # call the training routine using reasonable default parameters.
    model, history = train_model(
        data_dir="processed_data",           # Path to preprocessed, split dataset
        num_epochs=20,                       # Number of training epochs
        batch_size=32,                       # Mini-batch size
        learning_rate=0.001                  # Initial learning rate for Adam optimizer
    )

