import matplotlib.pyplot as plt


def visualize_xyy(x, y1, y2, title='', xlabel='', ylabel=''):
    # plot 2 lines with the same X-coordinate set
    plt.plot(x, y1, 'bo-', color='r', label='Loss')
    plt.plot(x, y2, 'bo-', color='g', label='Test score')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.show()