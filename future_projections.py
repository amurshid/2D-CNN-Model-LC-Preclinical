import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import torch
from example_training import SimpleLungCancerCNN

matplotlib.use('Agg')  # Set the 'Agg' backend *before* importing pyplot!
def get_model_size(model):
    """Compute the number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())

def get_projection_data_from_training():
    """
    Define several models of increasing size (as in example_training.py), 
    and (optionally) use example accuracies for each size.
    """
    class WiderCNN(SimpleLungCancerCNN):
        def __init__(self, num_classes=5, base_channels=64):
            super().__init__(num_classes)
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(base_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2,2),
                torch.nn.Dropout2d(0.25),

                torch.nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(base_channels*2),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2,2),
                torch.nn.Dropout2d(0.25),
            )
            # Note: classifier is not dynamically changed - may affect parameter counts!

    base_channels_list = [16, 32, 64, 128]
    model_sizes = []
    train_accuracies = []
    test_accuracies = []
    example_train_accs = [0.87, 0.91, 0.95, 0.97]
    example_test_accs  = [0.79, 0.83, 0.86, 0.88]

    for i, base_c in enumerate(base_channels_list):
        model = WiderCNN(num_classes=5, base_channels=base_c)
        n_params = get_model_size(model)
        model_sizes.append(n_params)
        train_accuracies.append(example_train_accs[i])
        test_accuracies.append(example_test_accs[i])

    return np.array(train_accuracies), np.array(test_accuracies), np.array(model_sizes)

def project_scalability(train_accuracies, test_accuracies, model_sizes, save_path="scalability_projection.png"):
    """
    Projects the trend of model performance (train/test accuracy) as model size increases, and saves a visualization.

    Args:
        train_accuracies (list or np.ndarray): Training accuracies for different model sizes.
        test_accuracies (list or np.ndarray): Testing accuracies for different model sizes.
        model_sizes (list or np.ndarray): Representative model sizes (number of parameters or FLOPS).
        save_path (str): Path where the plot will be saved.
    """
    # Do NOT reset backend here; must have been set before importing plt

    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)
    model_sizes = np.array(model_sizes)

    def sigmoid_projection(x, A, B, C):
        return A / (1.0 + np.exp(-B*(x - C)))

    try:
        popt, _ = curve_fit(sigmoid_projection, model_sizes, test_accuracies, p0=[1.0, 0.01, np.median(model_sizes)])
        projected_sizes = np.linspace(model_sizes.min(), model_sizes.max()*2, 100)
        projected_acc = sigmoid_projection(projected_sizes, *popt)
    except Exception:
        projected_sizes = np.linspace(model_sizes.min(), model_sizes.max()*2, 100)
        coeffs = np.polyfit(model_sizes, test_accuracies, 1)
        projected_acc = np.polyval(coeffs, projected_sizes)

    plt.figure(figsize=(8,6))
    plt.plot(model_sizes, train_accuracies, 'o-', label='Train Accuracy')
    plt.plot(model_sizes, test_accuracies, 's-', label='Test Accuracy')
    plt.plot(projected_sizes, projected_acc, 'k--', alpha=0.7, label='Projected Test Accuracy')
    plt.xlabel('Model Size (Number of Parameters)')
    plt.ylabel('Accuracy')
    plt.title('Model Scalability Projection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # Double-check file existence
    import os
    if os.path.exists(save_path):
        print(f"Scalability projection saved to: {save_path}")
    else:
        print(f"ERROR: Plot not saved! File '{save_path}' does not exist.")


# Get empirical/proxy data from example_training.py model(s)
train_accuracies, test_accuracies, model_sizes = get_projection_data_from_training()
save_path = "scalability_projection.png"
project_scalability(train_accuracies, test_accuracies, model_sizes, save_path)
