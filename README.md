# Project OCTOPUS

Projet de segmentation sémantique d'images sous-marines pour la détection de *Posidonia oceanica*.

Ce projet compare plusieurs architectures de deep learning, notamment VGG16-FCN8, U-Net et SegFormer, afin d'identifier les zones de posidonie dans des images acquises en milieu sous-marin.

## Objectifs

- Détecter et segmenter automatiquement la *Posidonia oceanica*.
- Comparer différentes architectures de segmentation sémantique.
- Étudier l'influence des prétraitements et des fonctions de perte.
- Évaluer les performances, l'incertitude et le temps d'inférence.
- Préparer une utilisation future à bord d'un véhicule sous-marin autonome.

## Modèles étudiés

- VGG16-FCN8
- U-Net
- SegFormer-B0
- SegFormer-B1
- SegFormer-B2
- Ensemble de modèles

## Fonctionnalités

- Entraînement et évaluation des modèles
- Validation simple et validation croisée
- Calibration du seuil de décision
- Estimation de l'incertitude
- Comparaison des temps d'inférence
- Application web de visualisation des prédictions

## Technologies

- Python
- PyTorch
- Transformers
- OpenCV
- NumPy
- Matplotlib
- Flask
