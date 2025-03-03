import dill
import torch
import numpy as np
import matplotlib.pyplot as plt
from USAVars import USAVars
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def retrieve_features(plottype, coord, lower, upper):
    if plottype == 'image':
        # Load the USAVars dataset
        dataset = USAVars(root="/share/usavars", download=False, split='train')  # Ensure path is correct

        # Store all images as flattened feature vectors
        image_list = []
        subsample_image_list = []

        for sample in tqdm(dataset):
            image = sample["image"]  # Assuming 'image' key contains pixel data
            image_flat = image.view(-1).numpy()  # Flatten image to 1D array
            image_list.append(image_flat)
            if lower <= sample[f'centroid_{coord}'].item() <= upper:
                subsample_image_list.append(image_flat)

        # Convert list to NumPy array (N, D) where N = number of images, D = flattened dimension
        X = np.array(image_list)
        X_subsample = np.array(subsample_image_list)

    elif plottype == 'torchgeo':
        with open("data/int/feature_matrices/CONTUS_UAR_torchgeo4096.pkl", "rb") as f:
            arrs = dill.load(f)

        X = arrs['X']
        latlon = arrs['latlon']

        if coord == 'lat':
            coords = latlon[:, 0]
        elif coord == 'lon':
            coords = latlon[:, 1]
        
        subsample_idxs = np.where((lower <= coords) & (coords <= upper))
        X_subsample = X[subsample_idxs]

    return X, X_subsample

def feature_pca_plot(plottype='image', coord='lon', lower=-130, upper=-115):

    X, X_subsample = retrieve_features(plottype, coord, lower, upper)

    pca = PCA(n_components=2).fit(X)
    pca_features = pca.transform(X)
    subsample_pca_features = pca.transform(X_subsample)

    plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.4, s=0.1, color='orangered', label='Image features')
    plt.scatter(subsample_pca_features[:, 0], subsample_pca_features[:,1], alpha=0.4, s=0.1, color='steelblue', label='Subsample features')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Projection of Image Features")

    lgnd = ax.legend(loc='lower right', scatterpoints=1)
    for handle in lgnd.legend_handles:
        handle.set_sizes([6.0])

    plt.tight_layout()
    plt.savefig("test.png")

def feature_tsne_plot(plottype='image', coord='lon', lower=-130, upper=-115):
    X, X_subsample = retrieve_features(plottype, coord, lower, upper)

    X = X[np.random.choice(len(X), size=10000, replace=False)]
    X_subsample = X_subsample[np.random.choice(len(X_subsample), size=10000*int(np.floor(len(X_subsample)/ len(X_subsample))), replace=False)]

    scaler = StandardScaler().fit(X) # Only if features have different scales
    X_scaled = scaler.transform(X)
    X_subsample_scaled = scaler.transform(X_subsample)

    # Step 2: Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit(X_scaled)
    X_tsne = tsne.fit_transform(X_scaled)
    X_subsample_tsne = tsne.fit_transform(X_subsample_scaled)

    # Step 3: Plot the t-SNE output
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.4, s=0.1, color='orangered', label='Image features')
    ax.scatter(X_subsample_tsne[:, 0], X_subsample_tsne[:,1], alpha=0.4, s=0.1, color='steelblue', label='Subsample features')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.suptitle("t-SNE")

    lgnd = ax.legend(loc='lower right', scatterpoints=1)
    for handle in lgnd.legend_handles:
        handle.set_sizes([6.0])

    fig.tight_layout()
    fig.savefig("test.png")

if __name__ == '__main__':
    feature_tsne_plot(plottype = 'torchgeo')
