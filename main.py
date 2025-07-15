import sys

import AppliedML.courselib.models.svm as svm
import AppliedML.courselib.models.nn as nn

from AppliedML.courselib.utils.metrics import accuracy, mean_squared_error
from AppliedML.courselib.utils.preprocessing import labels_encoding

from AppliedML.courselib.optimizers import GDOptimizer

import datagen.ConcentricBands as cb
import datagen.SeparatedBlobs as sb
import datagen.HalfMoons as hm
import datagen.ImageSampler as im
import datagen.Spirals as sp

from datagen.Postprocessors import *

from models.TunableRandomForest import TunableRandomForest

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from models.TunableModel import TunableModel


def get_optimizer(params):
    if params["activation"] == "ReLU":
        return GDOptimizer(learning_rate=1)
    else:
        return GDOptimizer(learning_rate=5)


def validator(params):
    if params["kernel"] == "rbf":
        return (
            params["sigma"] > 0 and params["degree"] == 0 and params["intercept"] == 0
        )
    elif params["kernel"] == "polynomial":
        return (
            params["sigma"] == 0 and params["degree"] > 0 and params["intercept"] >= 0
        )
    else:
        return False


if __name__ == "__main__":

    # params = {
    #     'kernel': ['rbf', 'polynomial'],
    #     'sigma': [0.025, 0.1],
    #     'degree': range(1),
    #     'intercept': range(3),
    # }

    params = {
        "widths": [(2, 10, 10, 2), (2, 100, 100, 2)],
        "activation": ["ReLU"],
        # 'activation': ['ReLU', 'Sigmoid'],
    }

    params["optimizer"] = get_optimizer

    tunable_model = TunableModel(
        model_class=nn.MLP,
        hyperparameters=params,
        validator=None,
    )

    # tunable_model = TunableModel(
    #     model_class=svm.BinaryKernelSVM,
    #     hyperparameters=params,
    #     validator=validator
    # )

    total_samples = 400
    train_test_split = 0.5

    num_train_samples = int(total_samples * train_test_split)

    # seed = 83703 - 0.02, 0.1
    # seed = 910666 - 0.025, 0.1

    iterations = 0
    while True:
        seed = np.random.randint(0, 1000000)
        # seed = 179350
        np.random.seed(seed)

        sampler = sp.Spirals(sp.SpiralConf(2, 0.1, 0.2, 1, 1))
        # sampler.add_postprocesser(LabelNoise(0.05))

        df = sampler.sample(total_samples)
        # df['label'] = df['label'].astype("category").cat.codes * 2 - 1
        # df['label'] = labels_encoding(df['label'].to_numpy())

        train_df = df[:num_train_samples]
        test_df = df[num_train_samples:]

        # models = tunable_model.fit(train_df[['x', 'y']].to_numpy(), train_df['label'].to_numpy())

        models = tunable_model.fit(
            train_df[["x", "y"]].to_numpy(),
            labels_encoding(train_df["label"].to_numpy()),
            training_params={
                "num_epochs": 10000,
                "batch_size": len(df[["label"]]),
                # 'compute_metrics': True,
                # 'metrics_dict': {'accuracy': accuracy, 'loss': mean_squared_error}
            },
        )

        accuracies = []
        for model, params, metrics in models:
            h_train = model(train_df[["x", "y"]].to_numpy())
            h_test = model(test_df[["x", "y"]].to_numpy())

            train_accuracy = accuracy(h_train, train_df["label"].to_numpy(), False)
            test_accuracy = accuracy(h_test, test_df["label"].to_numpy(), False)

            accuracies.append((train_accuracy, test_accuracy))

        print(
            f"Iteration {iterations}: Low: ({accuracies[0][0]:.2f}, {accuracies[0][1]:.2f}), High: ({accuracies[-1][0]:.2f}, {accuracies[-1][1]:.2f})"
        )
        iterations += 1

        # if accuracies[0][0] == 100 and accuracies[0][1] < 80:
        #     if accuracies[1][0] == 100 and accuracies[1][1] == 100:
        #         break
        break

    print(f"Found models with seed {seed} after {iterations} iterations.")

    # image = im.ImageSampler.open_image("smile.png")
    # image = np.vectorize(lambda x: 1 if x > 0 else 0)(image)
    #
    # pallete = {
    #     (1, 1, 1, 1): 0,  # White
    #     (1, 0, 0, 1): 1,  # Red
    #     (0, 1, 0, 1): 2,  # Green
    #     (0, 0, 1, 1): 3,  # Blue
    # }
    #
    # labels = {
    #     0: 'background',
    #     1: 'eye1',
    #     2: 'eye2',
    #     3: 'mouth'
    # }
    #
    # sampler = im.ImageSampler(pallete, labels=labels, image=image, random_seed=42)
    # sampler = sb.RandomSeparatedBlobs(2)
    # sampler = sp.Spirals(sp.SpiralConf(2, 0.05, 0.2, 1, 2))
    # sampler = cb.RandomConcentricBands(2, 0.2, 0.9)
    # sampler = hm.HalfMoons()

    # sampler.add_postprocesser(LabelNoise(0.01))
    # sampler.add_postprocesser(DomainShift(10, 10))

    # df = sampler.sample(total_samples)

    # df['label'] = labels_encoding(df['label'].to_numpy())
    # df['label'] = df['label'].astype("category").cat.codes * 2 - 1

    # train_df = df[:num_train_samples]
    # test_df = df[num_train_samples:]

    # plt.figure(figsize=(10, 10))
    # plt.scatter(x=df['x'], y=df['y'], c=df['label'].astype("category").cat.codes, cmap='viridis', edgecolor='k')
    # plt.show()

    # params = {
    #     'n_estimators': [1, 10, 50],
    #     'max_depth': [2, 10, 50],
    #     'min_samples_split': [2],
    #     'max_features': [None]
    # }
    #
    # tunable_model = TunableRandomForest(
    #     hyperparameters=params,
    #     validator=None
    # )

    # models = tunable_model.fit(train_df[['x', 'y']].to_numpy(), train_df['label'].to_numpy())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    x_min, x_max = df["x"].min() - 0.2, df["x"].max() + 0.2
    y_min, y_max = df["y"].min() - 0.2, df["y"].max() + 0.2
    x_list, y_list = np.meshgrid(
        np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
    )
    X_list = np.dstack([x_list, y_list])

    cmap = matplotlib.colors.ListedColormap(["red", "blue"])
    colors = df["label"].astype("category").cat.codes

    for ax, (model, params, metrics) in zip(axs.flat, models):
        # h_train = model.decision_function(train_df[['x', 'y']].to_numpy())
        # h_test = model.decision_function(test_df[['x', 'y']].to_numpy())
        h_train = model(train_df[["x", "y"]].to_numpy())
        h_test = model(test_df[["x", "y"]].to_numpy())

        train_accuracy = accuracy(h_train, train_df["label"].to_numpy(), False)
        test_accuracy = accuracy(h_test, test_df["label"].to_numpy(), False)

        # ax.set_title(f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}")
        # ax.set_title(f"Kernel: {params['kernel']}, Sigma: {params['sigma']:.1f}, Degree: {params['degree']}, Intercept: {params['intercept']}")

        ax.set_title(
            f"Width: {params['widths']} Train: {train_accuracy:.2f}%, Test: {test_accuracy:.2f}%"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set(aspect="equal")

        # Plot decision boundary
        h_list = model(X_list)

        ax.contourf(x_list, y_list, h_list, cmap=cmap, alpha=0.3)
        ax.scatter(df["x"], df["y"], c=colors, edgecolor="k")

    plt.tight_layout()
    plt.show()

    # gen = cb.RandomConcentricBands(2, 0.2, 0.5, False)
#     gen = hm.HalfMoons()
#
#     df = gen.sample(200)

# plt.figure(figsize=(10, 10))
# plt.scatter(df['x'], df['y'], c=df['label'].astype("category").cat.codes, cmap='viridis', edgecolor='k')
# plt.show()

#     params = {
#         'kernel': ['rbf', 'polynomial'],
#         'sigma': np.linspace(0, 1.0, 10),
#         'degree': range(1),
#         'intercept': range(3),
#     }
#
#     tunable_model = TunableModel(
#         model_class=svm.BinaryKernelSVM,
#         hyperparameters=params,
#         validator=validator
#     )
#
#     labels = np.unique(df['label'])
#     label_map = {label: i for i, label in enumerate(labels)}
#     df['label'] = df['label'].map(label_map).astype(int)
#
#     models = tunable_model.fit(
#         df[['x', 'y']].to_numpy(),
#         df['label'].to_numpy() * 2 - 1
#     )
#
#     print(f"Trained {len(models)} models with different hyperparameters.")
#
#     fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#     x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
#     y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5
#     x_list, y_list = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#     X_list = np.dstack([x_list, y_list])
#
#     cmap = matplotlib.colors.ListedColormap(['red', 'blue'])
#     colors = np.where(df['label'] * 2 - 1 < 0, 'red', 'blue')
#
#     for ax, (model, params, metrics) in zip(axs.flat, models):
#         ax.set_title(f"Kernel: {params['kernel']}, Sigma: {params['sigma']:.1f}, Degree: {params['degree']}, Intercept: {params['intercept']}")
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
#
#         # Plot decision boundary
#         h_list = np.where(model.decision_function(X_list) < 0, -1, 1)
#
#         ax.contourf(x_list, y_list, h_list, cmap=cmap, alpha=0.3)
#         ax.scatter(df['x'], df['y'], c=colors, edgecolor='k')
#
#     plt.tight_layout()
#     plt.show()

#     params = {
#         'widths': [(2, 10, 4), (2, 10, 10, 4), (2, 50, 10, 4), (2, 50, 50, 4),
#                    (2, 100, 4), (2, 100, 100, 4), (2, 500, 100, 4), (2, 500, 500, 4)],
#         'activation': ['ReLU'],
#         # 'activation': ['ReLU', 'Sigmoid'],
#     }
#
#     params['optimizer'] = get_optimizer
#
#     tunable_model = TunableModel(
#         model_class=nn.MLP,
#         hyperparameters=params,
#         validator=None,
#     )
#
#     models = tunable_model.fit(
#         df[['x', 'y']].to_numpy(),
#         labels_encoding(df['label'].to_numpy()),
#         training_params={'num_epochs': 10000, 'batch_size': len(df[['label']])},
#         metrics_dict={'compute_metrics': True, 'metrics_dict': {'accuracy': accuracy, 'loss': mean_squared_error}}
#     )
#
#     print(f"Trained {len(models)} models with different hyperparameters.")
#
#     fig, axs = plt.subplots(1, 4, figsize=(20, 10))
#
#     x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
#     y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5
#     x_list, y_list = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#     X_list = np.dstack([x_list, y_list])
#
#     cmap = matplotlib.colors.ListedColormap(['red', 'blue', 'green'])
#     colors = df['label'].astype("category").cat.codes
#
#     for ax, (model, params, metrics) in zip(axs.flat, models):
#         ax.set_title(f"Widths: {params['widths']}, Activation: {params['activation']}")
#
#         # ax.plot(range(len(metrics['loss'])), metrics['loss'], label='Loss')
#         # ax.plot(range(len(metrics['accuracy'])), metrics['accuracy'], label='Accuracy')
#
#         print(f"Final accuracy: {metrics['accuracy'][-1]:.4f}, Final loss: {metrics['loss'][-1]:.4f}")
#
#         # Plot decision boundary
#         h_list = model(X_list)
#
#         ax.contourf(x_list, y_list, h_list, cmap=cmap, alpha=0.3)
#         ax.scatter(df['x'], df['y'], c=colors, edgecolor='k')
#
#     plt.tight_layout()
#     plt.show()
