import matplotlib.pyplot as plt
import numpy as np


def visualize(x, lines, labels, title='', xlabel='', ylabel=''):
    # Plot multi lines with the same X-coordinate set

    list_color = ['r', 'g', 'y', 'b']
    for i in range(len(lines)):
        plt.plot(x, lines[i], 'bo-', color=list_color[i], label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.show()