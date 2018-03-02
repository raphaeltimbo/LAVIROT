import matplotlib.pyplot as plt


class Plot:
    pass


class BearingSealPlots(Plot):
    def __init__(self, element):
        self._element = element

        parameters = ['kxx', 'kyy', 'kxy', 'kyx',
                      'cxx', 'cyy', 'cxy', 'cyx']

        def plot(x, y, ax=None):
            if ax is None:
                ax = plt.gca()

            ax.plot(x, y)

            return ax

        for p in parameters:
            setattr(self, p, plot())



