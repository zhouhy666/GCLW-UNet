# WGCBA-Net
## Project Overview
WGCBA-Net is a wavelet-guided global-context and boundary-aware semantic segmentation network for industrial steel surface defects. It follows a U-Net encoder-decoder design and combines three paper-defined components: GC-ASPPM for multi-scale context aggregation, WACM for Haar-wavelet-guided channel/spatial recalibration, and Boundary-Constrained Dice (BCD) loss for contour-sensitive supervision. The paper reports 81.5% mIoU and 91.3% mAP on the NEU-seg benchmark.

Repository: https://github.com/zhouhy666/WGCBA-Net

## Core Commands
### Environment configuration
Please refer to requirements.txt.

### Dataset acquisition
NEU-Sge：https://github.com/DHW-Master/NEU_Seg
SSDD（Severstal Steel Defect Detection）:https://www.kaggle.com/c/severstal-steel-defect-detection

### Training the Model
```bash
# Single model training
python train.py --dataset pascal --model_type WGCBA_Net --batch-size 8 --loss-type dice_ce_boundary --epochs 100

# Specify GPUs
python train.py --dataset pascal --model_type WGCBA_Net --gpu-ids 0,1

# Resume training from checkpoint
python train.py --dataset pascal --model_type UNet --resume /path/to/checkpoint.pth.tar

# Validate the model
python val.py --model_type WGCBA_Net --resume /path/to/model_best.pth.tar

# Inference / Prediction
python predict.py --model_type WGCBA_Net --resume /path/to/model_best.pth.tar

##Environment Verification
```bash
python env_test.py
```

## Architecture Overview
### Directory Structure
```
WGCBA-Net/
├── blocks/              # Pluggable modules (Attention, Convolution improvements, etc.)
├── models/              # Model definitions
│   ├── unet_model.py    # Main model file (contains all improved models)
│   ├── unet_parts.py    # U-Net blocks plus WACM and GC-ASPPM
│   └── model_zoo.py     # Model registry
├── dataloaders/         # Data loading
├── utils/               # Utilities (loss, metrics, saver)
├── NEU_Seg_data/        # NEU-SEG dataset
├── train.py/val.py/predict.py  # Training/Validation/Inference scripts
└── ssddval.py           # Validation script for SSDD dataset
```

### Module Call Chain
1.blocks/: Contains reusable attention and convolution modules.
2.models/unet_model.py: Defines all UNet variant models, directly using modules via from blocks import *.
3.models/model_zoo.py: Unified registry MODEL_ZOO, retrieves model classes by string name.
4.models/unet_parts.py: Implements the paper-aligned WACM and GC-ASPPM modules.
5.train.py: Uses the --model_type argument to fetch the model class from globals() or MODEL_ZOO.

### Model Registration Process
Three-step workflow to add a new model:
1.Create a new module under blocks/ (optional).
2.Define the model class in models/unet_model.py.
3.Register it in the MODEL_ZOO dictionary within models/model_zoo.py.

### Data Flow
```
mypath.py (Path configuration)
  → dataloaders/__init__.py (make_data_loader factory function)
  → datasets/mydataset.py (Custom Dataset class)
```

### Training Pipeline
```
train.py → Trainer class
  → training(epoch): Mixed precision training (AMP) + tqdm progress bar
  → validation(epoch): Evaluator calculates mIoU/mPA/Dice
  → Saver: Saves checkpoints and TensorBoard logs
```

## Key Configurations

### Dataset Configuration
  - NEU_Seg_data/ (NEU-SEG)
  - SSDD_data/ (SSDD)

### Loss Functions
  - CrossEntropy (Default)
  - Focal Loss
  - Dice Loss
  - Boundary-Constrained Dice (`dice_ce_boundary`, weights 0.4/0.4/0.2)

### Learning Rate Scheduler
- poly (Default)
- step
- cos (Cosine annealing)

## Supported Models
- Check the MODEL_ZOO dictionary in models/model_zoo.py, which includes:
- `WGCBA_Net` (paper-aligned WGCBA-Net)
- UNet baseline and variants
- Comparison models (UNetPP, DeepLabV3Plus, SegNet, PSPNet, BiSeNet, OCNet, etc.)
