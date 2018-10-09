from __future__ import print_function

import os
import yaml
import argparse
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from data import DataLoader
from model import get_model

def scatter(x, labels, config):
    palette = np.array(sns.color_palette("hls", config["data"]["num_classes"]))

    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1], lw=0, s=40, alpha=0.1, c=palette[labels.astype(np.int)])

    for idx in range(config["data"]["num_classes"]):
        xtext, ytext = np.median(x[labels == idx, :], axis=0)
        txt = ax.text(xtext, ytext, str(idx), fontsize=20)

    plt.title("{} T-SNE".format(config["run-title"]))
    plt.savefig(os.path.join(config["paths"]["save"], "tsne.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    paths = config["paths"]
    data = config["data"]

    dataloader = DataLoader(data)
    dataloader.load()

    input_shape = (data["imsize"], data["imsize"], data["imchannel"])
    model = get_model(input_shape, config, top=False)

    model.load_weights(paths["load"], by_name=True)

    X_batch, y_batch = dataloader.get_random_batch(k = -1)

    #embeddings = X_batch.reshape(-1, 784) 
    embeddings = model.predict(X_batch, batch_size=config["train"]["batch-size"], verbose=1)

    tsne = TSNE(n_components=2, perplexity=30, verbose=1, n_iter=5000)
    tsne_embeds = tsne.fit_transform(embeddings)
    scatter(tsne_embeds, y_batch, config)