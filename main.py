import AppliedML.courselib.models.svm as svm
import AppliedML.courselib.models.nn as nn

from AppliedML.courselib.utils.metrics import accuracy, mean_squared_error
from AppliedML.courselib.utils.preprocessing import labels_encoding

from AppliedML.courselib.optimizers import GDOptimizer

import datagen.ConcentricBands as cc

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from models.TunableModel import TunableModel

gen = cc.ConcentricCircles(num_circles=3, num_samples_per_circle=100, variation=0.9)

df = gen.generate_sample()

plt.figure(figsize=(10, 10))
plt.scatter(df['x'], df['y'], c=df['numeric_label'], cmap='viridis', edgecolor='k')
plt.show()

params = {
    'kernel': ['rbf', 'polynomial'],
    'sigma': np.linspace(0, 1.0, 4),
    'degree': range(3),
    'intercept': range(3),
}

def validator(params):
    if params['kernel'] == 'rbf':
        return params['sigma'] > 0 and params['degree'] == 0 and params['intercept'] == 0
    elif params['kernel'] == 'polynomial':
        return params['sigma'] == 0 and params['degree'] > 0 and params['intercept'] >= 0
    else:
        return False
        
tunable_model = TunableModel(
    model_class=svm.BinaryKernelSVM,
    hyperparameters=params,
    validator=validator
)

# models = tunable_model.fit(
#     df[['x', 'y']].to_numpy(),
#     df['numeric_label'].to_numpy() * 2 - 1
# )
# 
# print(f"Trained {len(models)} models with different hyperparameters.")
# 
# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
# y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5
# x_list, y_list = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
# X_list = np.dstack([x_list, y_list])
# 
# cmap = matplotlib.colors.ListedColormap(['red', 'blue'])
# colors = np.where(df['numeric_label'] * 2 - 1 < 0, 'red', 'blue')
# 
# for ax, (model, params, metrics) in zip(axs.flat, models):
#     print(params)
#     ax.set_title(f"Kernel: {params['kernel']}, Sigma: {params['sigma']:.1f}, Degree: {params['degree']}, Intercept: {params['intercept']}")
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     
#     # Plot decision boundary
#     h_list = np.where(abs(model.decision_function(X_list)) <= 0.05, 1, -1)
#     
#     ax.contour(x_list, y_list, h_list, cmap=cmap, alpha=0.3)
#     ax.scatter(df['x'], df['y'], c=colors, edgecolor='k')
#     
# plt.tight_layout()
# plt.show()

params = {
    'widths': [(2, 10, 3), (2, 10, 10, 3), (2, 50, 10, 3), (2, 50, 50, 3)],
    'activation': ['ReLU', 'Sigmoid'],
}

tunable_model = TunableModel(
    model_class=nn.MLP,
    hyperparameters=params,
    validator=None,
    optimizer=GDOptimizer(learning_rate=1)
)

models = tunable_model.fit(
    df[['x', 'y']].to_numpy(),
    labels_encoding(df['numeric_label'].to_numpy()),
    training_params={'num_epochs': 10000, 'batch_size': len(df[['numeric_label']])},
    metrics_dict={'compute_metrics': True, 'metrics_dict': {'accuracy': accuracy, 'loss': mean_squared_error}}
)

print(f"Trained {len(models)} models with different hyperparameters.")

fig, axs = plt.subplots(2, 4, figsize=(20, 10))

x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5
x_list, y_list = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
X_list = np.dstack([x_list, y_list])

cmap = matplotlib.colors.ListedColormap(['red', 'blue', 'green'])
colors = df['numeric_label'].apply(lambda x: 'red' if x == 0 else ("blue" if x == 1 else 'green')).to_numpy()

for ax, (model, params, metrics) in zip(axs.flat, models):
    ax.set_title(f"Widths: {params['widths']}, Activation: {params['activation']}")
    
    # ax.plot(range(len(metrics['loss'])), metrics['loss'], label='Loss')
    # ax.plot(range(len(metrics['accuracy'])), metrics['accuracy'], label='Accuracy')
    
    # Plot decision boundary
    h_list = model(X_list)
    
    ax.contourf(x_list, y_list, h_list, cmap=cmap, alpha=0.3)
    ax.scatter(df['x'], df['y'], c=colors, edgecolor='k')
    
plt.tight_layout()
plt.show()