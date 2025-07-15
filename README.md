# Visualizing Overfitting and Model Complexity

This project explores how increasing model complexity influences decision boundaries and generalization. Using 2D datasets, we have examined how different models separate data in input space, and how this reflects underfitting, overfitting, and the bias–variance tradeoff.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Overview

The most important parts of this project are

- main.ipynb: the jupyter notebook containing the results
- datagen: a python package containing synthetic data generators for half moons, concentric bands, gaussian blobs, …
- models: a python package containing classes for testing models (perceptrons, kernel SVMs, …) with different parameters (in parallel)
- AppliedML: the repository of the course upon which this project builds upon
