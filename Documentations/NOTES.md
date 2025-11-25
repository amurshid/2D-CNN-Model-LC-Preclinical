# Terminal Outputs and Commands - Notes for Future Reference

This document contains verbatim terminal outputs and commands used during the CT scan preprocessing project setup.

---

## Package Installation

### Checking Python Version
```bash
python -c "import sys; print(sys.version)"
```

**Output:**
```
3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
```

---

### Installing PyTorch and torchvision
```bash
pip install torch torchvision
```

**Output:**
```
Collecting torch
  Using cached torch-2.9.0-cp310-cp310-win_amd64.whl.metadata (30 kB)
Collecting torchvision
  Using cached torchvision-0.24.0-cp310-cp310-win_amd64.whl.metadata (5.9 kB)
Collecting filelock (from torch)
  Using cached filelock-3.20.0-py3-none-any.whl.metadata (2.1 kB)
Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (4.10.0)
Requirement already satisfied: sympy>=1.13.3 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (3.4.2)
Requirement already satisfied: jinja2 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torch) (3.1.3)
Collecting fsspec>=0.8.5 (from torch)
  Using cached fsspec-2025.10.0-py3-none-any.whl.metadata (10 kB)
Requirement already satisfied: numpy in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torchvision) (1.24.3)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from torchvision) (10.0.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\azmay\appdata\local\packages\pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0\localcache\local-packages\python310\site-packages (from jinja2->torch) (2.1.5)
Using cached torch-2.9.0-cp310-cp310-win_amd64.whl (109.3 MB)
Using cached torchvision-0.24.0-cp310-cp310-win_amd64.whl (3.7 MB)
Using cached fsspec-2025.10.0-py3-none-any.whl (200 kB)
Using cached filelock-3.20.0-py3-none-any.whl (16 kB)
Installing collected packages: fsspec, filelock, torch, torchvision
  WARNING: The scripts torchfrtrace.exe and torchrun.exe are installed in 'C:\Users\azmay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts torchfrtrace.exe and torchrun.exe are installed in 'C:\Users\azmay\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is being used by another process: 'C:\\Users\\azmay\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages\\torchvision\\python310.dll'
Check the permissions.
```

**Note:** Despite the error, torch was successfully installed (verified in next step).

---

### Verifying PyTorch Installation
```bash
python -c "import torch; print('torch installed successfully')"
```

**Output:**
```
torch installed successfully
```

---

### Verifying torchvision Installation
```bash
python -c "import torchvision; print(f'torchvision installed successfully - version: {torchvision.__version__}')"
```

**Output:**
```
torchvision installed successfully - version: 0.24.0+cpu
```

---

### Checking All Required Packages
```bash
python check_and_install_packages.py
```

**Output:**
```
Checking for installed packages...
============================================================
[OK] torch - INSTALLED
[OK] torchvision - INSTALLED
[OK] numpy - INSTALLED
[OK] opencv-python (imported as cv2) - INSTALLED
[OK] Pillow (imported as PIL) - INSTALLED
[OK] scikit-learn (imported as sklearn) - INSTALLED
[OK] tqdm - INSTALLED
[OK] matplotlib - INSTALLED
[OK] seaborn - INSTALLED
============================================================

[SUCCESS] All required packages are already installed!
```

---

### Verifying All Imports
```bash
python -c "import torch; import torchvision; import numpy; import cv2; from PIL import Image; from sklearn.model_selection import train_test_split; import tqdm; import matplotlib; import seaborn; print('All imports successful!')"
```

**Output:**
```
All imports successful!
```

---

## Image Analysis

### Checking Sample Image Properties
```bash
python -c "from PIL import Image; import os; img_path = os.path.join('Lung Cancer Dataset', 'adenocarcinoma', os.listdir('Lung Cancer Dataset/adenocarcinoma')[0]); img = Image.open(img_path); print(f'Image size: {img.size}, Mode: {img.mode}')"
```

**Output:**
```
Image size: (315, 245), Mode: RGBA
```

**Note:** This confirmed that some images are RGBA format, which is handled by the preprocessing pipeline.

---

## Preprocessing Execution

### Correct Command to Run Preprocessing
```bash
python preprocess_ct_scans.py
```

**Note:** The correct filename is `preprocess_ct_scans.py` (not `preprocessing_ct_scan.py`).

**Alternative Command:**
```bash
python run_preprocessing.py
```

---

### Common Error - Incorrect Filename
```bash
python preprocessing_ct_scan.py
```

**Output:**
```
C:\Users\azmay\AppData\Local\Microsoft\WindowsApps\python.exe: can't open file 'C:\\CTSCAN-pre-processing\\preprocessing_ct_scan.py': [Errno 2] No such file or directory
```

**Fix:** Use the correct filename: `preprocess_ct_scans.py`

---

## Preprocessing Results

### Dataset Statistics (Expected After Preprocessing)

**Original Dataset:**
- adenocarcinoma: 337 PNG images
- Benign cases: 120 JPG images
- large cell carcinoma: 187 PNG images
- Normal cases: 631 images (428 JPG, 203 PNG)
- squamous cell carcinoma: 260 PNG images
- **Total: ~1,535 images**

**After Preprocessing (70/15/15 split):**
- **Train set**: ~1,074 images (70%)
- **Validation set**: ~230 images (15%)
- **Test set**: ~230 images (15%)

**Output Structure:**
```
processed_data/
├── train/
│   ├── adenocarcinoma/
│   ├── Benign cases/
│   ├── large cell carcinoma/
│   ├── Normal cases/
│   └── squamous cell carcinoma/
├── val/
│   └── [same class structure]
├── test/
│   └── [same class structure]
└── metadata.json
```


CT Scan Preprocessing Pipeline
============================================================

Analyzing dataset...
Total images: 1535
  adenocarcinoma: 337 images
  Benign cases: 120 images
  large cell carcinoma: 187 images
  Normal cases: 631 images
  squamous cell carcinoma: 260 images

Splitting dataset (train=0.7, val=0.15, test=0.15)...
Train: 1074 images
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 

Processing test set...
Processing test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231/231 [00:06<00:00, 33.55it/s] 

Preprocessing complete!
Successfully processed: 1535 images
Failed: 0 images
Processed images saved to: processed_data

Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 

Processing test set...
Processing test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231/231 [00:06<00:00, 33.55it/s] 

============================================================
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 

Processing test set...
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s] 
Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Validation: 230 images
Test: 231 images

Processing train set...
Validation: 230 images
Test: 231 images

Validation: 230 images
Test: 231 images
Validation: 230 images
Validation: 230 images
Validation: 230 images
Test: 231 images

Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Validation: 230 images
Test: 231 images

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]


Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]

Processing train set...
Processing train: 100%|████████████████████████████████| 1074/1074 [00:30<00:00, 35.10it/s]


Processing val set...
Processing val set...
Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s]

Processing test set...
Processing test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████Processing val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 230/230 [00:06<00:00, 34.04it/s]

Processing test set...
Processing test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231/231 [00:06<00:00, 33.55it/s]


Preprocessing complete!
Successfully processed: 1535 images
Failed: 0 images
Processed images saved to: processed_data
============================================================


---

## Verification Commands

### Verify Preprocessing Results
```bash
python verify_preprocessing.py
```

**Expected Output Format:**
```
======================================================================
PREPROCESSING VERIFICATION REPORT
======================================================================

📋 PREPROCESSING SETTINGS:
----------------------------------------------------------------------
  Image size: [224, 224]
  Normalized: True
  Grayscale: False

📊 ORIGINAL DATASET STATISTICS:
----------------------------------------------------------------------
  [Class counts from metadata]

📁 DATASET SPLITS:
----------------------------------------------------------------------
TRAIN SET:
  [Class distribution in train set]

VAL SET:
  [Class distribution in val set]

TEST SET:
  [Class distribution in test set]
----------------------------------------------------------------------

TRAIN SET:
  adenocarcinoma                :  236 images
  Benign cases                  :   84 images
  large cell carcinoma          :  131 images
  Normal cases                  :  441 images
  squamous cell carcinoma       :  182 images
  TOTAL                         : 1074 images

VAL SET:
  adenocarcinoma                :   50 images
  Benign cases                  :   18 images
  large cell carcinoma          :   28 images
  Normal cases                  :   95 images
  squamous cell carcinoma       :   39 images
  TOTAL                         :  230 images

TEST SET:
  adenocarcinoma                :   51 images
  Benign cases                  :   18 images
  large cell carcinoma          :   28 images
  Normal cases                  :   95 images
  squamous cell carcinoma       :   39 images
  TOTAL                         :  231 images

⚖️  CLASS DISTRIBUTION ANALYSIS:
----------------------------------------------------------------------
  [Imbalance analysis]

🔍 TESTING DATA LOADER:
----------------------------------------------------------------------
  ✓ Data loaders created successfully
  ✓ Number of classes: 5
  ✓ Classes: [class names]
  ✓ Train batches: [number]
  ✓ Validation batches: [number]
  ✓ Test batches: [number]

✅ PREPROCESSING VERIFICATION COMPLETE
   Your dataset is ready for training!
======================================================================
```

IMAGE FORMAT DISTRIBUTION ANALYSIS
======================================================================

📊 ORIGINAL DATASET - FORMAT DISTRIBUTION BY CLASS:
----------------------------------------------------------------------

adenocarcinoma:
  Total: 337 images
  PNG: 337 (100.0%)
  JPG: 0 (0.0%)
  ⚠️  WARNING: All images are PNG - format-class correlation detected!

Benign cases:
  Total: 120 images
  PNG: 0 (0.0%)
  JPG: 120 (100.0%)
  ⚠️  WARNING: All images are JPG - format-class correlation detected!

large cell carcinoma:
  Total: 187 images
  PNG: 187 (100.0%)
  JPG: 0 (0.0%)
  ⚠️  WARNING: All images are PNG - format-class correlation detected!

Normal cases:
  Total: 631 images
  PNG: 203 (32.2%)
  JPG: 428 (67.8%)

squamous cell carcinoma:
  Total: 260 images
  PNG: 260 (100.0%)
  JPG: 0 (0.0%)
  ⚠️  WARNING: All images are PNG - format-class correlation detected!

📈 OVERALL STATISTICS:
----------------------------------------------------------------------
Total images: 1535
PNG: 987 (64.3%)
JPG: 548 (35.7%)
---

## Troubleshooting Notes

### Issue: File Access Error During Installation
**Error:**
```
ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is being used by another process
```

**Solution:** Close any Python processes or IDEs that might be using the packages, then retry installation.

---

### Issue: ModuleNotFoundError
**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:** Install packages using `pip install -r requirements.txt`

---

### Issue: UnicodeEncodeError in Windows Console
**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0
```

**Solution:** Use ASCII characters instead of Unicode symbols in print statements for Windows compatibility.

---

## Important File Names

- Preprocessing script: `preprocess_ct_scans.py`
- Helper script: `run_preprocessing.py`
- Verification script: `verify_preprocessing.py`
- Data loader: `data_loader.py`
- Training example: `example_training.py`
- Requirements: `requirements.txt`

---

## Package Versions Installed

- torch: 2.9.0
- torchvision: 0.24.0+cpu
- numpy: 1.24.3
- opencv-python: 4.8.1.78
- Pillow: 10.0.0
- scikit-learn: 1.4.1.post1
- tqdm: 4.65.0
- matplotlib: 3.8.0
- seaborn: 0.13.2

---

## Next Steps After Preprocessing

1. Verify preprocessing: `python verify_preprocessing.py`
2. Start training: `python example_training.py`
3. Monitor training progress and adjust hyperparameters as needed

---

## Date Created
Document created for future reference to maintain consistency in preprocessing pipeline setup and troubleshooting.

## TRAINING PROGRESS

Epoch 18/20
------------------------------------------------------------
Training:   0%|                                                                                                           | 0/34 [00:00<?, ?it/s]Error loading image processed_data\train\Normal cases\Normal cases (135).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (21).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (64).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:   3%|██▉                                                                                                | 1/34 [00:02<01:25,  2.60s/it]Error loading image processed_data\train\Benign cases\Benign cases  (67).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:   6%|█████▊                                                                                             | 2/34 [00:04<01:19,  2.47s/it]Error loading image processed_data\train\Normal cases\Normal cases (196).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (11).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (154).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  15%|██████████████▌                                                                                    | 5/34 [00:12<01:11,  2.46s/it]Error loading image processed_data\train\Normal cases\Normal cases (37).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (55).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  18%|█████████████████▍                                                                                 | 6/34 [00:14<01:09,  2.47s/it]Error loading image processed_data\train\Normal cases\Normal cases (310).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (194).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  21%|████████████████████▍                                                                              | 7/34 [00:17<01:06,  2.44s/it]Error loading image processed_data\train\Normal cases\Normal cases (117).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Benign cases\Benign cases  (47).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  24%|███████████████████████▎                                                                           | 8/34 [00:19<01:03,  2.44s/it]Error loading image processed_data\train\Normal cases\Normal cases (248).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (249).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  26%|██████████████████████████▏                                                                        | 9/34 [00:22<01:00,  2.43s/it]Error loading image processed_data\train\Normal cases\Normal cases (376).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (154).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  29%|████████████████████████████▊                                                                     | 10/34 [00:24<00:57,  2.41s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (228).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (106).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  32%|███████████████████████████████▋                                                                  | 11/34 [00:26<00:55,  2.40s/it]Error loading image processed_data\train\Normal cases\Normal cases (346).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  35%|██████████████████████████████████▌                                                               | 12/34 [00:29<00:52,  2.41s/it]Error loading image processed_data\train\Normal cases\Normal cases (177).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  38%|█████████████████████████████████████▍                                                            | 13/34 [00:31<00:50,  2.42s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (19).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Training:  41%|████████████████████████████████████████▎                                                         | 14/34 [00:34<00:48,  2.44s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (34).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  44%|███████████████████████████████████████████▏                                                      | 15/34 [00:36<00:46,  2.44s/it]Error loading image processed_data\train\Normal cases\Normal cases (24).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (287).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  47%|██████████████████████████████████████████████                                                    | 16/34 [00:39<00:43,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (27).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\Normal cases\Normal cases (73).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (284).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  53%|███████████████████████████████████████████████████▉                                              | 18/34 [00:43<00:38,  2.43s/it]Error loading image processed_data\train\Benign cases\Benign cases  (64).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (93).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  56%|██████████████████████████████████████████████████████▊                                           | 19/34 [00:46<00:36,  2.41s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (290).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (38).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  59%|█████████████████████████████████████████████████████████▋                                        | 20/34 [00:48<00:33,  2.40s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (86).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Training:  68%|██████████████████████████████████████████████████████████████████▎                               | 23/34 [00:56<00:26,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (225).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (73).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  71%|█████████████████████████████████████████████████████████████████████▏                            | 24/34 [00:58<00:24,  2.44s/it]Error loading image processed_data\train\Normal cases\Normal cases (144).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  76%|██████████████████████████████████████████████████████████████████████████▉                       | 26/34 [01:03<00:19,  2.50s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (193).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\Normal cases\Normal cases (223).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  79%|█████████████████████████████████████████████████████████████████████████████▊                    | 27/34 [01:06<00:17,  2.48s/it]Error loading image processed_data\train\Normal cases\Normal cases (398).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (413).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (94).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (44).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  82%|████████████████████████████████████████████████████████████████████████████████▋                 | 28/34 [01:08<00:14,  2.48s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (270).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  88%|██████████████████████████████████████████████████████████████████████████████████████▍           | 30/34 [01:13<00:09,  2.49s/it]Error loading image processed_data\train\Normal cases\Normal cases (414).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  91%|█████████████████████████████████████████████████████████████████████████████████████████▎        | 31/34 [01:15<00:07,  2.47s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (260).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (19).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (125).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  94%|████████████████████████████████████████████████████████████████████████████████████████████▏     | 32/34 [01:18<00:04,  2.49s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (115).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (115).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Benign cases\Benign cases  (70).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  97%|███████████████████████████████████████████████████████████████████████████████████████████████   | 33/34 [01:21<00:02,  2.51s/it]Error loading image processed_data\train\Normal cases\Normal cases (193).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:22<00:00,  2.42s/it]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.23it/s] 
Train Loss: 0.6125, Train Acc: 74.58%
Val Loss: 0.5434, Val Acc: 75.22%
Learning Rate: 0.001000
✓ New best validation accuracy: 75.22%

Epoch 19/20
------------------------------------------------------------
Training:   0%|                                                                                                           | 0/34 [00:00<?, ?it/s]Error loading image processed_data\train\Normal cases\Normal cases (233).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (214).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Error loading image processed_data\train\Normal cases\Normal cases (232).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:   6%|█████▊                                                                                             | 2/34 [00:04<01:17,  2.41s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (110).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (17).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (108).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:   9%|████████▋                                                                                          | 3/34 [00:07<01:14,  2.40s/it]Error loading image processed_data\train\Normal cases\Normal cases (107).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  12%|███████████▋                                                                                       | 4/34 [00:09<01:12,  2.41s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (47).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (259).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  15%|██████████████▌                                                                                    | 5/34 [00:12<01:09,  2.40s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (39).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Training:  18%|█████████████████▍                                                                                 | 6/34 [00:14<01:07,  2.40s/it]Error loading image processed_data\train\Normal cases\Normal cases (16).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  21%|████████████████████▍                                                                              | 7/34 [00:16<01:05,  2.42s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (22).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\Benign cases\Benign cases  (72).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  24%|███████████████████████▎                                                                           | 8/34 [00:19<01:02,  2.42s/it]Error loading image processed_data\train\Normal cases\Normal cases (68).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (62).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (203).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  26%|██████████████████████████▏                                                                        | 9/34 [00:21<01:01,  2.48s/it]Error loading image processed_data\train\Normal cases\Normal cases (40).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (67).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  29%|████████████████████████████▊                                                                     | 10/34 [00:24<00:59,  2.49s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (276).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\Benign cases\Benign cases  (47).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  32%|███████████████████████████████▋                                                                  | 11/34 [00:27<00:57,  2.52s/it]Error loading image processed_data\train\Normal cases\Normal cases (109).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  38%|█████████████████████████████████████▍                                                            | 13/34 [00:31<00:52,  2.48s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (68).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (8).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  41%|████████████████████████████████████████▎                                                         | 14/34 [00:34<00:49,  2.47s/it]Error loading image processed_data\train\Normal cases\Normal cases (33).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (194).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (78).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  44%|███████████████████████████████████████████▏                                                      | 15/34 [00:36<00:46,  2.45s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (217).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (185).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  47%|██████████████████████████████████████████████                                                    | 16/34 [00:39<00:44,  2.45s/it]Error loading image processed_data\train\Normal cases\Normal cases (185).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (161).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (209).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (272).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  50%|█████████████████████████████████████████████████                                                 | 17/34 [00:41<00:41,  2.45s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (49).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (179).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  53%|███████████████████████████████████████████████████▉                                              | 18/34 [00:44<00:39,  2.48s/it]Error loading image processed_data\train\Normal cases\Normal cases (154).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (121).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (98).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  56%|██████████████████████████████████████████████████████▊                                           | 19/34 [00:46<00:37,  2.49s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (11).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Benign cases\Benign cases  (66).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (218).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  59%|█████████████████████████████████████████████████████████▋                                        | 20/34 [00:49<00:34,  2.46s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (79).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (161).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  62%|████████████████████████████████████████████████████████████▌                                     | 21/34 [00:51<00:31,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (55).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Training:  65%|███████████████████████████████████████████████████████████████▍                                  | 22/34 [00:54<00:29,  2.48s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (121).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (11).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  68%|██████████████████████████████████████████████████████████████████▎                               | 23/34 [00:56<00:27,  2.49s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (182).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (2).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  74%|████████████████████████████████████████████████████████████████████████                          | 25/34 [01:01<00:22,  2.51s/it]Error loading image processed_data\train\Benign cases\Benign cases  (105).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  76%|██████████████████████████████████████████████████████████████████████████▉                       | 26/34 [01:04<00:19,  2.49s/it]Error loading image processed_data\train\Normal cases\Normal cases (14).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (95).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  79%|█████████████████████████████████████████████████████████████████████████████▊                    | 27/34 [01:06<00:17,  2.50s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (113).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (160).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (326).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (37).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  82%|████████████████████████████████████████████████████████████████████████████████▋                 | 28/34 [01:08<00:14,  2.46s/it]Error loading image processed_data\train\Normal cases\Normal cases (424).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (114).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (53).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  85%|███████████████████████████████████████████████████████████████████████████████████▌              | 29/34 [01:11<00:12,  2.48s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (216).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (124).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (119).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (368).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (227).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  88%|██████████████████████████████████████████████████████████████████████████████████████▍           | 30/34 [01:14<00:10,  2.53s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (258).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\Normal cases\Normal cases (334).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  91%|█████████████████████████████████████████████████████████████████████████████████████████▎        | 31/34 [01:16<00:07,  2.55s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (22).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (311).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  94%|████████████████████████████████████████████████████████████████████████████████████████████▏     | 32/34 [01:19<00:04,  2.50s/it]Error loading image processed_data\train\Normal cases\Normal cases (32).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:23<00:00,  2.45s/it]
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.18it/s] 
Train Loss: 0.6110, Train Acc: 74.67%
Val Loss: 0.5414, Val Acc: 74.78%
Learning Rate: 0.001000

Epoch 20/20
------------------------------------------------------------
Training:   0%|                                                                                                           | 0/34 [00:00<?, ?it/s]Error loading image processed_data\train\Normal cases\Normal cases (136).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:   3%|██▉                                                                                                | 1/34 [00:02<01:19,  2.42s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (250).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\Normal cases\Normal cases (109).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (78).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (270).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:   6%|█████▊                                                                                             | 2/34 [00:04<01:17,  2.42s/it]Error loading image processed_data\train\Normal cases\Normal cases (14).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (256).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:   9%|████████▋                                                                                          | 3/34 [00:07<01:14,  2.40s/it]Error loading image processed_data\train\Normal cases\Normal cases (279).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (126).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Error loading image processed_data\train\Normal cases\Normal cases (332).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  12%|███████████▋                                                                                       | 4/34 [00:09<01:12,  2.41s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (47).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  15%|██████████████▌                                                                                    | 5/34 [00:12<01:10,  2.42s/it]Error loading image processed_data\train\Normal cases\Normal cases (55).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (382).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  18%|█████████████████▍                                                                                 | 6/34 [00:14<01:07,  2.43s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (268).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  21%|████████████████████▍                                                                              | 7/34 [00:16<01:05,  2.42s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (94).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (332).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Error loading image processed_data\train\Normal cases\Normal cases (200).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  24%|███████████████████████▎                                                                           | 8/34 [00:19<01:02,  2.42s/it]Error loading image processed_data\train\Benign cases\Benign cases  (116).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\Normal cases\Normal cases (171).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  26%|██████████████████████████▏                                                                        | 9/34 [00:21<01:00,  2.40s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (177).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  35%|██████████████████████████████████▌                                                               | 12/34 [00:29<00:53,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (165).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (102).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (163).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  38%|█████████████████████████████████████▍                                                            | 13/34 [00:31<00:50,  2.43s/it]Error loading image processed_data\train\Normal cases\Normal cases (321).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (43).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  41%|████████████████████████████████████████▎                                                         | 14/34 [00:33<00:48,  2.43s/it]Error loading image processed_data\train\Normal cases\Normal cases (175).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (69).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (329).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  44%|███████████████████████████████████████████▏                                                      | 15/34 [00:36<00:46,  2.44s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (121).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (16).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (102).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (52).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (39).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  47%|██████████████████████████████████████████████                                                    | 16/34 [00:38<00:43,  2.44s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (171).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (85).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  50%|█████████████████████████████████████████████████                                                 | 17/34 [00:41<00:41,  2.45s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (5).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)      
Training:  53%|███████████████████████████████████████████████████▉                                              | 18/34 [00:43<00:39,  2.50s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (74).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  56%|██████████████████████████████████████████████████████▊                                           | 19/34 [00:46<00:37,  2.48s/it]Error loading image processed_data\train\Benign cases\Benign cases  (79).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (81).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  59%|█████████████████████████████████████████████████████████▋                                        | 20/34 [00:48<00:34,  2.48s/it]Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (80).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (19).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (163).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  62%|████████████████████████████████████████████████████████████▌                                     | 21/34 [00:51<00:33,  2.55s/it]Error loading image processed_data\train\Normal cases\Normal cases (25).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (337).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  65%|███████████████████████████████████████████████████████████████▍                                  | 22/34 [00:54<00:30,  2.54s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (56).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\Normal cases\Normal cases (12).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (18).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  68%|██████████████████████████████████████████████████████████████████▎                               | 23/34 [00:56<00:27,  2.52s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (114).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\Normal cases\Normal cases (272).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (47).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  71%|█████████████████████████████████████████████████████████████████████▏                            | 24/34 [00:58<00:25,  2.50s/it]Error loading image processed_data\train\Normal cases\Normal cases (120).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Benign cases\Benign cases  (41).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  74%|████████████████████████████████████████████████████████████████████████                          | 25/34 [01:01<00:22,  2.48s/it]Error loading image processed_data\train\Normal cases\Normal cases (55).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (41).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (103).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  76%|██████████████████████████████████████████████████████████████████████████▉                       | 26/34 [01:03<00:19,  2.49s/it]Error loading image processed_data\train\Normal cases\Normal cases (398).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (101).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (7).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Training:  79%|█████████████████████████████████████████████████████████████████████████████▊                    | 27/34 [01:06<00:17,  2.46s/it]Error loading image processed_data\train\Normal cases\Normal cases (368).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Error loading image processed_data\train\Normal cases\Normal cases (395).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  82%|████████████████████████████████████████████████████████████████████████████████▋                 | 28/34 [01:08<00:14,  2.45s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (257).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (16).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)    
Training:  85%|███████████████████████████████████████████████████████████████████████████████████▌              | 29/34 [01:11<00:12,  2.45s/it]Error loading image processed_data\train\Normal cases\Normal cases (323).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)        
Training:  88%|██████████████████████████████████████████████████████████████████████████████████████▍           | 30/34 [01:13<00:09,  2.44s/it]Error loading image processed_data\train\Normal cases\Normal cases (71).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (116).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (108).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training:  91%|█████████████████████████████████████████████████████████████████████████████████████████▎        | 31/34 [01:16<00:07,  2.44s/it]Error loading image processed_data\train\squamous cell carcinoma\squamous cell carcinoma (70).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\large cell carcinoma\large cell carcinoma (151).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
Error loading image processed_data\train\Normal cases\Normal cases (127).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  94%|████████████████████████████████████████████████████████████████████████████████████████████▏     | 32/34 [01:18<00:04,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (55).png: At least one stride in the given numpy array is negative, and tensError loading image processed_data\train\Normal cases\Normal cases (127).jpg: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  94%|████████████████████████████████████████████████████████████████████████████████████████████▏     | 32/34 [01:18<00:04,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (55).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (227).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:22<00:00,  2.42s/it] 
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.22it/s] 
Train Loss: 0.5934, Train Acc: 76.26%
s with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)       
Training:  94%|████████████████████████████████████████████████████████████████████████████████████████████▏     | 32/34 [01:18<00:04,  2.44s/it]Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (55).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (227).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:22<00:00,  2.42s/it] 
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.22it/s] 
Train Loss: 0.5934, Train Acc: 76.26%
Val Loss: 0.6463, Val Acc: 71.74%
Learning Rate: 0.001000
ors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)     
Error loading image processed_data\train\adenocarcinoma\adenocarcinoma (227).png: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:22<00:00,  2.42s/it] 
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.22it/s] 
Train Loss: 0.5934, Train Acc: 76.26%
Val Loss: 0.6463, Val Acc: 71.74%
Learning Rate: 0.001000

============================================================
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:22<00:00,  2.42s/it] 
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.22it/s] 
Train Loss: 0.5934, Train Acc: 76.26%
Val Loss: 0.6463, Val Acc: 71.74%
Learning Rate: 0.001000

============================================================
Training completed!
Val Loss: 0.6463, Val Acc: 71.74%
Learning Rate: 0.001000

============================================================
Training completed!
Best validation accuracy: 75.22%

============================================================
Training completed!
Best validation accuracy: 75.22%

Training completed!
Best validation accuracy: 75.22%

Evaluating on test set...
Best validation accuracy: 75.22%

Evaluating on test set...

Evaluating on test set...
Evaluating on test set...
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.15it/s] 
Test Loss: 0.5392, Test Acc: 77.92%
Test Loss: 0.5392, Test Acc: 77.92%


============================================================
Training completed!
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [01:24<00:00,  2.48s/it] 
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.16it/s] 
Train Loss: 0.5917, Train Acc: 74.86%
Val Loss: 0.5586, Val Acc: 76.96%
Learning Rate: 0.001000

============================================================
Training completed!
Train Loss: 0.5917, Train Acc: 74.86%
Val Loss: 0.5586, Val Acc: 76.96%
Learning Rate: 0.001000

============================================================
Training completed!
Val Loss: 0.5586, Val Acc: 76.96%
Learning Rate: 0.001000

============================================================
Training completed!
Learning Rate: 0.001000

============================================================
Training completed!

============================================================
Training completed!
============================================================
Training completed!
Training completed!
Best validation accuracy: 77.83%
Best validation accuracy: 77.83%

Evaluating on test set...
Evaluating on test set...
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.15it/s] 
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:06<00:00,  1.15it/s] 
Test Loss: 0.4912, Test Acc: 80.95%
Test Loss: 0.4912, Test Acc: 80.95%

