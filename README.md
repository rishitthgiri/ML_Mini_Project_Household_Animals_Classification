# Household Animals Classification Using Deep Learning
UE24CS352A Machine Learning - Mini Project (Problem 7)
**Team 7** | PES1UG24CS376, PES1UG24CS409 | Section G

Binary classification of household animals (cat vs dog) with PyTorch. Based on the CS229 (Stanford, 2020) report by Lei Lin.

## Pipeline
1. VGG-style CNN trained from scratch (baseline)
2. VGG-16 transfer learning (ImageNet weights, frozen conv layers, new classifier)
3. VGG-16 fine-tuning (block5 unfrozen)
4. Visualisation: intermediate feature maps and Grad-CAM heatmaps

**Dataset:** Kaggle Dogs vs Cats (~23k images), loaded automatically from the Hugging Face mirror `microsoft/cats_vs_dogs` (no API key needed). Split 80/10/10 with seed 42.

## Run on Google Colab
1. Runtime > Change runtime type > **T4 GPU**.
2. In a Colab cell:
```python
!git clone https://github.com/rishitthgiri/ML_Mini_Project_Household_Animals_Classification.git
%cd ML_Mini_Project_Household_Animals_Classification
!pip install -q -r requirements.txt
!python train.py            # full run (~30-40 min on T4)
!python make_docs.py        # builds writeup.pdf and slides.pptx from the results
```
(Private repo: use a GitHub personal access token in the clone URL, or upload the files to Colab manually.)

Quick test (a few minutes): `!python train.py --subset 4000 --epochs_scratch 3 --epochs_head 2 --epochs_ft 2`

## Outputs (in `outputs/`)
`results.json`, `curves.png`, `confusion_matrix.png`, `activations.png`, `gradcam.png`, `vgg16_finetuned.pth`

## Files
- `train.py` - data, models, training, evaluation, visualisations
- `make_docs.py` - generates the 2-page PDF write-up and slide deck
- `requirements.txt`
