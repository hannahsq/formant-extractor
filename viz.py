# viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import umap

sns.set(style="whitegrid")

def plot_layerwise_accuracy(results, title="Layerwise Probe Accuracy"):
    layers = [r['layer'] for r in results]
    accs = [r['accuracy'] for r in results]
    plt.figure(figsize=(8,4))
    plt.plot(layers, accs, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0,1)
    plt.show()

def plot_pca_embeddings(X, labels, title="PCA of Embeddings", n_components=2):
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=labels, palette="tab10", s=40)
    plt.title(title)
    plt.show()

def plot_umap_embeddings(X, labels, n_neighbors=15, min_dist=0.1, metric='cosine'):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    Z = reducer.fit_transform(X)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=labels, palette="tab10", s=40)
    plt.title("UMAP of Embeddings")
    plt.show()

def plot_trajectory(embeddings, title="Embedding Trajectory"):
    """
    embeddings: (time_steps, dim) numpy array
    project to 2D with PCA and plot path
    """
    pca = PCA(n_components=2)
    Z = pca.fit_transform(embeddings)
    plt.figure(figsize=(6,4))
    plt.plot(Z[:,0], Z[:,1], '-o', markersize=3)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def plot_formant_r2(results):
    layers = [r['layer'] for r in results]
    r2s = [r['r2'] for r in results]

    plt.figure(figsize=(8,4))
    plt.plot(layers, r2s, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("R² (Formant Prediction)")
    plt.title("Layerwise Formant Regression")
    plt.ylim(0, 1)
    plt.show(block=False)


def plot_formant_r2_per_formant(results):
    layers = [r["layer"] for r in results]
    r2s = np.array([r["r2"] for r in results])  # shape (n_layers, 4)

    plt.figure(figsize=(10, 5))
    for f_idx, label in enumerate(["F1", "F2", "F3", "F4"]):
        plt.plot(layers, r2s[:, f_idx], marker="o", label=label)

    plt.xlabel("Layer")
    plt.ylabel("R² (Formant Prediction)")
    plt.title("Layerwise Formant Regression by Formant")
    plt.legend()
    plt.ylim(0, 1)
    plt.show(block=False)


def print_eval_report(results):
    print("\n=== Multi‑Head Evaluation ===")
    print(f"Vowel accuracy: {results['vowel_accuracy']:.3f}\n")

    for i, (mse, r2) in enumerate(zip(results["formant_mse"], results["formant_r2"])):
        print(f"Formant F{i+1}:")
        print(f"   MSE: {mse:.3f}")
        print(f"   R²:  {r2:.3f}\n")