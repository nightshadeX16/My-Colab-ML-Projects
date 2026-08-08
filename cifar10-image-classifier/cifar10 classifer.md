# CIFAR-10 Image Classifier

A convolutional neural network trained from scratch to classify images across the 10 CIFAR-10 categories, with data augmentation and a full per-class evaluation.

## Overview

This project builds and trains a CNN on the CIFAR-10 dataset, a standard benchmark of 60,000 32x32 color images across 10 object classes. Rather than reporting a single accuracy number, the model is evaluated per-class to see where it performs well and where it struggles.

## Dataset

- CIFAR-10: 50,000 training images, 10,000 test images (32x32 RGB)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Loaded from the original pickled batch files and reshaped into standard image tensors

## Approach

1. **Preprocessing** — unpickled and combined all 5 training batches (50,000 images total), reshaped from flat vectors into (32, 32, 3) image tensors, and normalized pixel values.
2. **Data Augmentation** — used Keras' `ImageDataGenerator` during training to improve generalization on the relatively small image size.
3. **Model Architecture** — a CNN with three convolutional blocks (32, 64, and 64 filters respectively, each with ReLU activation), followed by a dense layer and a softmax output over the 10 classes.
4. **Training** — trained for 50 epochs on the full training set, with the held-out 10,000-image test set used as validation data during training.
5. **Evaluation** — generated a full classification report (precision, recall, F1-score per class) on the test set, rather than relying on overall accuracy alone.

## Results (test set, 10,000 images)

Overall accuracy: **78%** | Macro-average F1-score: **0.78**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| airplane | 0.80 | 0.81 | 0.81 |
| automobile | 0.87 | 0.91 | 0.89 |
| bird | 0.85 | 0.56 | 0.67 |
| cat | 0.67 | 0.59 | 0.63 |
| deer | 0.81 | 0.68 | 0.74 |
| dog | 0.64 | 0.78 | 0.70 |
| frog | 0.76 | 0.88 | 0.82 |
| horse | 0.80 | 0.86 | 0.83 |
| ship | 0.92 | 0.86 | 0.89 |
| truck | 0.77 | 0.92 | 0.84 |

## Observations

Performance is strongest on classes with more distinctive shapes and less visual overlap (ship, automobile, truck), while classes that are visually similar to one another perform worse — notably cat (0.63 F1) and bird (0.67 F1), which are often confused with visually similar animal classes like dog and deer. This kind of per-class breakdown is more useful than a single accuracy figure because it shows exactly where the model needs improvement, rather than hiding weak classes behind a decent overall average.

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn (classification metrics)
- NumPy
- Matplotlib / Seaborn
- Google Colab

## Key Takeaways

- Data augmentation helped the model generalize better on a relatively small 32x32 image size.
- Per-class evaluation revealed confusions between visually similar categories that overall accuracy alone would have hidden.
- A relatively simple 3-block CNN can reach a solid ~78% accuracy on CIFAR-10 without transfer learning, though modern architectures or pretrained backbones would likely close the gap on the harder classes.
