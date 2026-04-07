# YOLO26-SVAT-Road-Waste-Detection
Official PyTorch implementation of **YOLO26-SVAT** for road waste small-target detection, from the paper:
> *Visual Computing-Based Road Cleanliness Assessment with Enhanced Small-Target Detection and Dynamic Risk Weighting*

---

## 📖 Overview
This repository contains the training, inference, and evaluation code for the **YOLO26-SVAT** model, an enhanced real-time object detector optimized for small-target road waste detection. The model integrates key improvements including SIoU loss, Varifocal Loss, ATSS label assignment, and label smoothing to significantly boost small-target detection performance without additional computational overhead, making it suitable for intelligent road sweeping applications.

---

## ⚙️ Environment Requirements
### Hardware
- NVIDIA RTX 5060Ti GPU (8GB+ VRAM recommended)
- Intel Core i7-14700 CPU
- 16GB+ RAM

### Software
- Ubuntu 20.04 LTS (or similar Linux distribution)
- Python 3.8+
- PyTorch 1.10.0+
- CUDA 11.3+
- Full dependency list is available in `pyproject.toml`

---

## 📦 Installation
1. Clone this repository:
```bash
git clone https://github.com/chenzihua-1/YOLO26-SVAT-Road-Waste-Detection.git
cd YOLO26-SVAT-Road-Waste-Detection

Install the required dependencies:
pip install -e .
# Alternative: pip install -r requirements.txt

📂 Dataset
We use a self-constructed road waste dataset for model training and evaluation:
Total images: 7537
Train/Val/Test split: 6029 / 753 / 755
Annotation format: COCO JSON
Dataset details, category definitions, and collection process are fully described in the paper.
Note: The dataset is not included in this repository due to size constraints; all experimental results are fully reproducible with the provided code and configuration.

🚀 Model Training
To train the YOLO26-SVAT model from scratch, run the following command:

python train.py \
  --img 640 \
  --batch 64 \
  --epochs 500 \
  --data data.yaml \
  --weights yolo26n.pt

🔍 Inference & Evaluation
1. Evaluate Model Performance
To evaluate the trained model on the test set and get standard detection metrics (mAP, Precision, Recall):
python test.py \
  --weights runs/train/exp/weights/best.pt \
  --data data.yaml \
  --img 640 \
  --conf 0.25 \
  --iou 0.45

2. Run Inference on Custom Images/Videos
To run inference on your own images or videos and save visualization results:
python visualize.py \
  --source path/to/your/image_or_video \
  --weights runs/train/exp/weights/best.pt \
  --conf 0.25 \
  --save-txt \
  --save-conf
📊 Experimental Results
The YOLO26-SVAT model achieves state-of-the-art performance on our road waste dataset:
Metric	Value
mAP@0.5	0.89
mAP@0.5:0.95	0.60
Detailed ablation studies, comparison with baseline detectors (YOLOv8, YOLOv11, etc.), and robustness analysis are provided in the paper.

📝 Citation
If you use this code or model in your research, please cite our paper:
@software{chenzihua2026yolo26svat,
  author={Chen, Zihua},
  title={YOLO26-SVAT-Road-Waste-Detection},
  year={2026},
  url={https://github.com/chenzihua-1/YOLO26-SVAT-Road-Waste-Detection},
  license={MIT}
}

📄 License
This project is licensed under the MIT License - see the file for full details.

Copyright
© 2026 Zihua Chen . All rights reserved.

