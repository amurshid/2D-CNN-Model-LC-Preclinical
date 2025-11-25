# How to Train the Model Using example_training.py

## Quick Start

### Step 1: Verify Prerequisites

Make sure you have:
1. ✅ Preprocessed data in `processed_data/` directory
2. ✅ All dependencies installed
3. ✅ Python 3.8+ installed

### Step 2: Install Dependencies

If you haven't already, install required packages:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch` - PyTorch for deep learning
- `torchvision` - Vision utilities
- `numpy` - Numerical operations
- `opencv-python` - Image processing
- `Pillow` - Image loading
- `scikit-learn` - Data splitting utilities
- `tqdm` - Progress bars
- `matplotlib` - Plotting training curves

### Step 3: Run Training

Simply execute the training script:

```bash
python example_training.py
```

That's it! The script will:
- Load your preprocessed data
- Create data loaders for train/val/test sets
- Initialize the CNN model
- Train for 20 epochs
- Save the best model as `best_model.pth`
- Generate training history plots

---

## Detailed Training Process

### What Happens During Training

1. **Data Loading** (automatic)
   - Loads images from `processed_data/train/`
   - Loads images from `processed_data/val/`
   - Loads images from `processed_data/test/`
   - Applies data augmentation to training set

2. **Model Initialization**
   - Creates `SimpleLungCancerCNN` with 5 output classes
   - Moves model to GPU (if available) or CPU
   - Initializes optimizer (Adam) and learning rate scheduler

3. **Training Loop** (20 epochs by default)
   - For each epoch:
     - Trains on all training batches
     - Validates on validation set
     - Adjusts learning rate if validation loss plateaus
     - Saves best model based on validation accuracy

4. **Final Evaluation**
   - Loads best model weights
   - Evaluates on test set
   - Reports final test accuracy

5. **Outputs**
   - `best_model.pth` - Saved model weights
   - `training_history.png` - Training curves plot

---

## Training Parameters

### Default Settings

The script uses these default parameters:

```python
data_dir="processed_data"      # Location of preprocessed data
num_epochs=20                  # Number of training epochs
batch_size=32                  # Images per batch
learning_rate=0.001            # Initial learning rate
```

### Customizing Training Parameters

You can modify the training script or call `train_model()` directly:

```python
from example_training import train_model

model, history = train_model(
    data_dir="processed_data",
    num_epochs=50,              # Train for more epochs
    batch_size=16,              # Smaller batches (if GPU memory is limited)
    learning_rate=0.0001        # Lower learning rate for fine-tuning
)
```

### Parameter Recommendations

**For GPU Training:**
```python
batch_size=32      # Good default
num_epochs=30-50   # More epochs often help
learning_rate=0.001
```

**For CPU Training:**
```python
batch_size=16      # Smaller to avoid memory issues
num_epochs=20      # Fewer epochs (slower on CPU)
learning_rate=0.001
```

**For Small Dataset (like yours):**
```python
batch_size=16      # Smaller batches help with small datasets
num_epochs=30-40   # More epochs to learn from limited data
learning_rate=0.0005  # Slightly lower to avoid overfitting
```

---

## Expected Output

### Console Output

You'll see output like this:

```
Using device: cuda  # or cpu
Loading data...
Loaded 1074 images from train split
Loaded 230 images from val split
Loaded 231 images from test split
Classes: ['adenocarcinoma', 'Benign cases', 'large cell carcinoma', 'Normal cases', 'squamous cell carcinoma']
Number of classes: 5

Starting training...
============================================================

Epoch 1/20
------------------------------------------------------------
Training: 100%|████████████| 34/34 [00:45<00:00,  1.32s/it]
Validating: 100%|██████████| 8/8 [00:05<00:00,  1.45s/it]
Train Loss: 1.2345, Train Acc: 45.23%
Val Loss: 1.1234, Val Acc: 52.17%
Learning Rate: 0.001000
✓ New best validation accuracy: 52.17%

Epoch 2/20
...
```

### Output Files

After training completes, you'll have:

1. **`best_model.pth`**
   - PyTorch model state dictionary
   - Contains the best model weights (highest validation accuracy)
   - Can be loaded later for inference

2. **`training_history.png`**
   - Two plots showing:
     - Training and validation loss over epochs
     - Training and validation accuracy over epochs

---

## Monitoring Training

### What to Watch For

**Good Signs:**
- ✅ Training loss decreases steadily
- ✅ Validation loss decreases (or stays stable)
- ✅ Validation accuracy increases
- ✅ Training and validation curves are close (not overfitting)

**Warning Signs:**
- ⚠️ Training loss decreases but validation loss increases → **Overfitting**
  - Solution: More dropout, less epochs, more data augmentation
- ⚠️ Both losses plateau early → **Underfitting**
  - Solution: More epochs, higher learning rate, more model capacity
- ⚠️ Large gap between train and val accuracy → **Overfitting**
  - Solution: More regularization, less model complexity

### Expected Performance

Based on your dataset size (~1,535 images):

- **Training accuracy**: 80-95% (may overfit)
- **Validation accuracy**: 70-85%
- **Test accuracy**: 65-80%

**Note**: These are realistic expectations. Higher accuracy would require more data or transfer learning.

---

## Troubleshooting

### Common Issues

#### 1. **Out of Memory Error**

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
batch_size=16  # or even 8

# Or reduce image size in preprocessing
img_size=(128, 128)  # instead of (224, 224)
```

#### 2. **Data Not Found**

**Error**: `FileNotFoundError` or `No images found`

**Solution**: Make sure you've run preprocessing first:
```bash
python preprocess_ct_scans.py
```

#### 3. **Slow Training on CPU**

**Issue**: Training is very slow

**Solutions**:
- Use GPU if available (CUDA)
- Reduce batch size
- Reduce number of epochs for testing
- Use smaller image size

#### 4. **Import Errors**

**Error**: `ModuleNotFoundError`

**Solution**: Install missing dependencies:
```bash
pip install -r requirements.txt
```

#### 5. **Model Not Improving**

**Issue**: Accuracy stays low or doesn't improve

**Solutions**:
- Train for more epochs (30-50)
- Adjust learning rate (try 0.0005 or 0.0001)
- Check if data is loaded correctly
- Verify labels are correct

---

## Advanced Usage

### Resuming Training

To resume from a saved checkpoint:

```python
import torch
from example_training import SimpleLungCancerCNN, train_model

# Load saved model
model = SimpleLungCancerCNN(num_classes=5)
model.load_state_dict(torch.load('best_model.pth'))

# Continue training (modify train_model to accept initial weights)
```

### Using Different Optimizers

Modify the training script:

```python
# Instead of Adam, try SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Or AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Adding Class Weights (for imbalanced data)

```python
from data_loader import CTScanDataset
import torch

# Calculate class weights
train_dataset = CTScanDataset("processed_data", split='train')
class_weights = train_dataset.get_class_weights()
weights_tensor = torch.FloatTensor([class_weights[i] for i in range(5)])

# Use weighted loss
criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
```

### Training with Transfer Learning

For better performance, use a pre-trained model:

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)
# Modify final layer for 5 classes
model.fc = nn.Linear(model.fc.in_features, 5)
```

---

## Training Time Estimates

**On CPU:**
- Per epoch: ~5-10 minutes
- Full training (20 epochs): ~2-3 hours

**On GPU (CUDA):**
- Per epoch: ~30-60 seconds
- Full training (20 epochs): ~15-20 minutes

**Note**: Times vary based on hardware and batch size.

---

## Next Steps After Training

1. **Evaluate the model**
   - Check `training_history.png` for training curves
   - Review test accuracy

2. **Use the model for inference**
   ```python
   import torch
   from example_training import SimpleLungCancerCNN
   
   model = SimpleLungCancerCNN(num_classes=5)
   model.load_state_dict(torch.load('best_model.pth'))
   model.eval()
   # Use for predictions...
   ```

3. **Improve the model**
   - Try transfer learning
   - Add class weights
   - Adjust hyperparameters
   - Collect more data

---

## Quick Reference

### Basic Training Command
```bash
python example_training.py
```

### Check if GPU is Available
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU available
```

### Verify Data is Ready
```bash
# Check processed_data directory structure
ls processed_data/train/
ls processed_data/val/
ls processed_data/test/
```

### Monitor GPU Usage (if using GPU)
```bash
# On Linux/Mac
watch -n 1 nvidia-smi

# On Windows (PowerShell)
# Use Task Manager or GPU monitoring tools
```

---

## Summary

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run preprocessing** (if not done): `python preprocess_ct_scans.py`
3. **Train model**: `python example_training.py`
4. **Check outputs**: `best_model.pth` and `training_history.png`
5. **Monitor training**: Watch for overfitting/underfitting
6. **Adjust parameters**: Modify script for better performance

That's it! Your model will train automatically and save the best version.

