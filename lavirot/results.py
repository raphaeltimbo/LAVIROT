import pickle
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

    def __reduce__(self):
        # TODO add documentation explaining reduce, setstate and save.
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self._new_attributes,)

        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self._new_attributes = state[-1]
        for k, v in self._new_attributes.items():
            setattr(self, k, v)
        super().__setstate__(state[0:-1])

    def save(self, file):
        with open(file, mode='wb') as f:
            pickle.dump(self, f)

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

        if units == 'm':
            ax0.set_ylabel('Amplitude $(m)$')
        elif units == 'mic-pk-pk':
            ax0.set_ylabel('Amplitude $(\mu pk-pk)$')
        else:
            ax0.set_ylabel('Amplitude $(dB)$')

        ax1.set_ylabel('Phase')
        ax1.set_xlabel('Frequency (rad/s)')

        return ax0, ax1

    def plot_freq_response_grid(self, outs, inps, ax=None, **kwargs):
        # TODO function not tested after being moved from rotor.py
        # TODO check if this can be integrated to the plot function
        """Plot frequency response.

        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        outs : list
            List with the desired outputs.
        inps : list
            List with the desired outputs.
        modes : list
            List with the modes that will be used to construct
            the frequency response plot.

        ax : array with matplotlib.axes, optional
            Matplotlib axes array created with plt.subplots.
            It needs to have a shape of (2*inputs, outputs).
        Returns
        -------
        ax : array with matplotlib.axes, optional
            Matplotlib axes array created with plt.subplots.

        Examples
        --------
        >>> m0, m1 = 1, 1
        >>> c0, c1, c2 = 1, 1, 1
        >>> k0, k1, k2 = 1e3, 1e3, 1e3
        >>> M = np.array([[m0, 0],
        ...               [0, m1]])
        >>> C = np.array([[c0+c1, -c2],
        ...               [-c1, c2+c2]])
        >>> K = np.array([[k0+k1, -k2],
        ...               [-k1, k2+k2]])
        >>> sys = VibeSystem(M, C, K) # create the system
        >>> # plot frequency response for inputs at [0, 1]
        >>> # and outputs at [0, 1]
        >>> sys.plot_freq_response_grid(outs=[0, 1], inps=[0, 1])
        array([[<matplotlib.axes._...
        """
        if ax is None:
            fig, ax = plt.subplots(len(inps) * 2, len(outs),
                                   sharex=True,
                                   figsize=(4 * len(outs), 3 * len(inps)))
            fig.subplots_adjust(hspace=0.001, wspace=0.25)

        if len(outs) > 1:
            for i, out in enumerate(outs):
                for j, inp in enumerate(inps):
                    self.plot(out, inp,
                              ax0=ax[2 * i, j],
                              ax1=ax[2 * i + 1, j], **kwargs,)
        else:
            for i, inp in enumerate(inps):
                self.plot(outs[0], inp,
                          ax0=ax[2 * i],
                          ax1=ax[2 * i + 1], **kwargs,)

        return ax
