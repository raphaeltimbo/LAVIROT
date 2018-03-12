import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Results(np.ndarray):
    """Class used to store results and provide plots.

    This class subclasses np.ndarray to provide additional info and a plot
    method to the calculated results from Rotor.

    Metadata about the results should be stored on info as a dictionary to be
    used on plot configurations and so on.

    Additional attributes can be passed as a dictionary in new_attributes kwarg.

    """
    def __new__(cls, input_array, new_attributes=None):
        obj = np.asarray(input_array).view(cls)

        # TODO evaluate if new_attributes is useful. Slicing may by a problem
        for k, v in new_attributes.items():
            setattr(obj, k, v)

        # save new attributes names to create them on array finalize
        obj._new_attributes = new_attributes

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        try:
            for k, v in obj._new_attributes.items():
                setattr(self, k, getattr(obj, k, v))
        except AttributeError:
            return

    def plot(self, *args, **kwargs):
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

        speed_range = self.speed_range

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

                whirl_mask = (whirl_i == whirl_dir)
                if whirl_mask.shape[0] == 0:
                    continue
                else:
                    im = ax.scatter(speed_range[whirl_mask], wd_i[whirl_mask],
                                    c=log_dec_i[whirl_mask],
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


class FrequencyResponseResults(Results):
    def plot(self, inp, out, ax0=None, ax1=None, units='m',
             **kwargs):
        """Plot frequency response.

        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax0 : matplotlib.axes, optional
            Matplotlib axes where the amplitude will be plotted.
            If None creates a new.
        ax1 : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with amplitude plot.
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.

        Examples
        --------
        """
        if ax0 is None or ax1 is None:
            fig, ax = plt.subplots(2)
            if ax0 is not None:
                _, ax1 = ax
            if ax1 is not None:
                ax0, _ = ax
            else:
                ax0, ax1 = ax
        # TODO add option to select plot units
        omega = self.omega
        magdb = self[:, :, :, 0]
        phase = self[:, :, :, 1]

        ax0.plot(omega, magdb[inp, out, :], **kwargs)
        ax1.plot(omega, phase[inp, out, :], **kwargs)
        for ax in [ax0, ax1]:
            ax.set_xlim(0, max(omega))
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(prune='lower'))
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(prune='upper'))

        ax0.text(.9, .9, 'Output %s' % inp,
                 horizontalalignment='center',
                 transform=ax0.transAxes)
        ax0.text(.9, .7, 'Input %s' % out,
                 horizontalalignment='center',
                 transform=ax0.transAxes)

        if units == 'm':
            ax0.set_ylabel('Amplitude $(m)$')
        else:
            ax0.set_ylabel('Amplitude $(dB)$')
        ax1.set_ylabel('Phase')
        ax1.set_xlabel('Frequency (rad/s)')

        return ax0, ax1

