# 🐒 Monkey Species Classifier & Detection using Swin Transformers

This repository contains a complete pipeline for classifying monkey species using Swin Transformers and visualizing simulated bounding box detections with confidence thresholds. The project includes training scripts, visualization plots, and simulated object detection with explanation support using LLMs (Large Language Models).

## 📂 Dataset

The dataset is organized into 10 classes:
- `n0` to `n9` representing species such as:
  - Mantled Howler
  - Patas Monkey
  - Bald Uakari
  - Japanese Macaque
  - Pygmy Marmoset
  - White-headed Capuchin
  - Silvery Marmoset
  - Common Squirrel Monkey
  - Black-headed Night Monkey
  - Nilgiri Langur


## 🧠 Model

We use **Swin Transformer** (`swin_tiny_patch4_window7_224`) for fine-tuning on the monkey dataset. The model is trained with PyTorch and supports GPU acceleration.

## 🔁 Training

```bash
python train_swin.py --train_dir ./seai_images/training --val_dir ./seai_images/validation --epochs 10 --save_model model_best.pt
After training, the following plots are generated:

Training vs Validation Accuracy

Confidence vs F1 Score

Confidence vs Recall Score

Learning Rate over Epochs
 

You can run the plotting script using: plot_metrics()


REQUIREMENTS:
Install via:

pip install -r requirements.txt


📄 License
This project is licensed under the MIT License.
