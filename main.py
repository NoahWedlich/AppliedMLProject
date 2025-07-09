import AppliedML.courselib.models.svm as svm
import datagen.ConcentricCircles as cc

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

gen = cc.ConcentricCircles(num_circles=2, num_samples_per_circle=100, variation=0.5)

df = gen.generate_sample()

sigmas = np.linspace(0.1, 1.0, 16)
models = []

for sigma in sigmas:
    model = svm.BinaryKernelSVM(kernel='rbf', sigma=sigma)
    print(f"Training model with sigma={sigma:.5f}")
    model.fit(df[['x', 'y']].to_numpy(), df['numeric_label'].to_numpy() * 2 - 1)
    models.append((sigma, model))
    
x_plots = round(len(sigmas) ** 0.5)
y_plots = int(np.ceil(len(sigmas) / x_plots))

fig, axs = plt.subplots(x_plots, y_plots, figsize=(15, 15))

x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5

x_list, y_list = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
X_list = np.dstack([x_list, y_list])

cmap = matplotlib.colors.ListedColormap(['red', 'blue'])
colors = np.where(df['numeric_label'] * 2 - 1 < 0, 'red', 'blue')

for ax, (sigma, model) in zip(axs.flat, models):
    ax.set_title(f"Sigma = {sigma:.5f}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Plot decision boundary
    # h_list = model(X_list)
    h_list = np.where(abs(model.decision_function(X_list)) <= 0.05, 1, -1)
    
    ax.contour(x_list, y_list, h_list, cmap=cmap, alpha=0.3)
    ax.scatter(df['x'], df['y'], c=colors, edgecolor='k')

plt.tight_layout()
plt.show()