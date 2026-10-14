# Few-Shot Image Classification: Backbone Benchmarking in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sasidharreddy-cu/Few-Shot-Learning-Project/blob/main/few_shot_backbone_benchmark.ipynb)

How well do ImageNet-pretrained CNNs recognize new categories when you have only **1, 5, 10 or 20 labeled images per class**? We benchmarked **ConvNeXt-Base, ResNet-50 and VGG-16** on three datasets of increasing difficulty: **Animals, Flowers and Insects**.

Final project for CSCI 5922 (Neural Networks & Deep Learning), University of Colorado Boulder, by Sasidhar Reddy Gangi Reddy and Cannon Palmer.

## Results

Macro-averaged accuracy on all held-out images. **Bold** = best model for that dataset and shot count.

| Dataset | Model | 1-shot | 5-shot | 10-shot | 20-shot |
|---|---|---:|---:|---:|---:|
| Animals | ConvNeXt-Base | **67.50%** | **81.84%** | **85.69%** | **89.22%** |
| Animals | ResNet-50 | 42.66% | 70.08% | 76.84% | 81.39% |
| Animals | VGG-16 | 10.90% | 15.64% | 16.69% | 16.08% |
| Flowers | ConvNeXt-Base | 31.82% | **66.67%** | **79.45%** | **87.14%** |
| Flowers | ResNet-50 | **32.32%** | 61.71% | 69.39% | 78.12% |
| Flowers | VGG-16 | 12.08% | 21.98% | 27.38% | 32.51% |
| Insects | ConvNeXt-Base | **11.95%** | **21.10%** | **26.35%** | **33.63%** |
| Insects | ResNet-50 | 8.30% | 18.68% | 24.98% | 31.00% |
| Insects | VGG-16 | 4.81% | 9.74% | 6.94% | 8.24% |

<p>
  <img src="results/animals_accuracy.png" width="32%" alt="Animals: accuracy vs samples per class">
  <img src="results/flowers_accuracy.png" width="32%" alt="Flowers: accuracy vs samples per class">
  <img src="results/insects_accuracy.png" width="32%" alt="Insects: accuracy vs samples per class">
</p>

## Key findings

- **ConvNeXt-Base was the best backbone in 11 of 12 settings.** It reached 89.2% on Animals and 87.1% on Flowers with only 20 images per class. ResNet-50 edged it out only on 1-shot Flowers (32.3% vs 31.8%).
- **A few more examples help a lot.** Going from 1 to 20 images per class raised ConvNeXt's accuracy by 21.7 points on Animals, 55.3 points on Flowers and 21.7 points on Insects.
- **Insects is hard for every model** (33.6% at best). It has 102 fine-grained pest classes, and none of their names appear among ImageNet's 1,000 classes (exact name match), so the pretrained features transfer poorly.
- **VGG-16 lagged far behind** (32.5% at best). Unlike the other two models, its entire three-layer fully connected head (about 120M parameters) was left trainable. That is a likely cause of overfitting on so few images.

## Method

- **Backbones:** torchvision ImageNet-pretrained ResNet-50, VGG-16 and ConvNeXt-Base. Convolutional features are frozen, and the final layer is replaced with one sized for the target classes.
- **Few-shot split:** for each class, *n* random images (*n* = 1, 5, 10, 20) are used for training, and all remaining images are used for testing.
- **Training:** Adam (learning rate 1e-3), batch size 32, 10 epochs, cross-entropy loss, 128×128 inputs with random horizontal flips, on a Colab GPU.
- **Evaluation:** macro-averaged accuracy, precision and recall, plus per-class accuracy (torchmetrics).
- **Datasets:** Animals, Flowers, and Insects (OpenML dataset 44326: 102 classes, about 75,000 images).

## Repository

| Path | Contents |
|---|---|
| `few_shot_backbone_benchmark.ipynb` | Full code with outputs: data preparation, training, evaluation, plots |
| `results/` | Accuracy vs. samples-per-class plots for each dataset |

## Running it

Open the notebook in Colab with the button above and select a GPU runtime (**Runtime → Change runtime type → T4 GPU**). The notebook reads the image datasets from Google Drive (`MyDrive/Colab Notebooks/Datasets/`), so to rerun training, place the datasets there or update the paths in the data cells. The saved outputs and plots can be viewed without rerunning anything.

## Tech

Python · PyTorch · torchvision · torchmetrics · scikit-learn · pandas · matplotlib · Google Colab
