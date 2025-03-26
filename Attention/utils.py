import numpy as np
import matplotlib.pyplot as plt

def visualize_tensor(tensors, words):
    tensors = tensors.detach().numpy()
    for i, tensor in enumerate(tensors):
        x = tensor[0]
        y = tensor[1]
        plt.scatter(x, y)
        plt.text(x, y, words[i], fontsize=9)
    plt.title("self attention Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()