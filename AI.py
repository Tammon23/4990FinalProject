from random import sample
from sklearn.manifold import TSNE
from PyQt6 import QtWidgets

import matplotlib

from Util import NNException

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PointPlotter(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(PointPlotter, self).__init__(fig)

        self.setMinimumSize(800, 800)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.model = None

    def set_model(self, model):
        self.model = model

    def TSNEPlot(self, perplexity=40, num_tokens_to_plot=100, use_random_sample=False, useLastN=False, words=None):
        if self.model is None:
            return

        self.axes.cla()
        "Creates and TSNE model and plots it"
        if num_tokens_to_plot >= 0:
            if use_random_sample:
                labels = sample(list(self.model.key_to_index), num_tokens_to_plot)
            elif words is not None:
                labels = list(words.intersection(set(self.model.key_to_index)))
                if len(labels) <= perplexity:
                    raise NNException(
                        f"Number of words given that exist in the trained corpus ( {len(labels)} ) must be greater than perplexity!")
            elif useLastN:
                labels = list(self.model.key_to_index)[-num_tokens_to_plot:]
            else:
                labels = list(self.model.key_to_index)[:num_tokens_to_plot]
        else:
            labels = list(self.model.key_to_index)

        tokens = self.model[labels]
        tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23, learning_rate=200)
        tokens_tsne = tsne.fit_transform(tokens)
        x, y = zip(*tokens_tsne)

        for i in range(len(x)):
            self.axes.scatter(x[i], y[i])
            self.axes.annotate(labels[i],
                               xy=(x[i], y[i]),
                               xytext=(5, 2),
                               textcoords='offset points',
                               ha='right',
                               va='bottom')
        self.draw()
