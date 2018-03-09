import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Results(np.ndarray):
    """Class used to store results and provide plots.

    This class subclasses np.ndarray to provide additional info and a plot
    method to the calculated results from Rotor.

    Metadata about the results should be stored on info as a dictionary to be
    used on plot configurations and so on.

    """
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    def plot(self):
        raise NotImplementedError


class CampbellResults(Results):
    def plot(self, harmonics=[1], fig=None, ax=None):
        """Plot campbell results.

        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        fig : matplotlib figure, optional
            Figure to insert axes with log_dec colorbar.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        """

        # results for campbell is an array with [speed_range, wd/log_dec/whirl]

        if fig is None and ax is None:
            fig, ax = plt.subplots()

        speed_range = self.info['speed_range']

        wd = self[:, :, 0]
        log_dec = self[:, :, 1]
        whirl = self[:, :, 2]

        for mark, whirl_dir in zip(['^', 'o', 'v'],
                                   [0., 0.5, 1.]):
            for i in range(wd.shape[1]):
                # get only forward
                wd_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]
                if whirl_i[whirl_i == whirl_dir].shape[0] == 0:
                    continue
                else:
                    im = ax.scatter(speed_range, wd_i[whirl_i == whirl_dir],
                                    c=log_dec_i[whirl_i == whirl_dir],
                                    marker=mark, cmap='RdBu', vmin=0.1,
                                    vmax=2., s=20, alpha=0.5)

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('log dec')
        cbar.solids.set_edgecolor("face")

        label_color = cbar.cmap(cbar.vmax, alpha=0.3)
        forward_label = mpl.lines.Line2D([], [], marker='^', lw=0,
                                         color=label_color,
                                         label='Forward')
        backward_label = mpl.lines.Line2D([], [], marker='v', lw=0,
                                          color=label_color,
                                          label='Backward')
        mixed_label = mpl.lines.Line2D([], [], marker='o', lw=0,
                                       color=label_color,
                                       label='Mixed')

        legend = plt.legend(
            handles=[forward_label, backward_label, mixed_label], loc=2)

        ax.add_artist(legend)

        ax.set_xlabel('Rotor speed ($rad/s$)')
        ax.set_ylabel('Damped natural frequencies ($rad/s$)')

        return ax


