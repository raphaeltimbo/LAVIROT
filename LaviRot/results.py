r"""
This module contains :py:meth:`LaviRot.results` with functions
to evaluate results and functions to create plots
"""
# TODO detail the results docstring
# TODO not possible to return figures. Return axs instead.

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from LaviRot.rotor import rotor_example


__all__ = ["plot_rotor",
           "MAC",
           "MAC_modes",
           "campbell",
           "bearing_parameters",
           "plot_freq_response",
           "plot_time_response"]

# set style and colors
plt.style.use('seaborn-white')
plt.style.use({
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'axes.linewidth': 0.1,
    'grid.color': '.9',
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.framealpha': 0.2
    })

_orig_rc_params = mpl.rcParams.copy()

c_pal = {'red': '#C93C3C',
         'blue': '#0760BA',
         'green': '#2ECC71',
         'dark blue': '#07325E',
         'purple': '#A349C6',
         'grey': '#2D2D2D',
         'green2': '#08A4AF'}



def campbell(rotor, speed_rad, freqs=6, mult=[1], plot=True, ax=None):
    r"""Calculates the Campbell diagram.

    This function will calculate the damped natural frequencies
    for a speed range.

    Parameters
    ----------
    rotor: Rotor instance
        Rotor instance that will be used for calculating the
        Campbell diagram.
    speed_rad: array
        Array with the speed range in rad/s.
    freqs: int, optional
        Number of frequencies that will be calculated.
        Default is 6.
    mult: list, optional
        List withe the harmonics to be plotted.
        The default is to plot 1x.
    plot: bool, optional
        If the campbell will be plotted.
        If plot=False, points for the Campbell will be returned.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.

    Returns
    -------
    points: array
        Array with the natural frequencies corresponding to each speed
         of the speed_rad array. It will be returned if plot=False.
    ax : matplotlib axes
        Returns the axes object with the plot.

    Examples
    --------
    >>> rotor1 = rotor_example()
    >>> speed = np.linspace(0, 400, 101)
    >>> camp = campbell(rotor1, speed, plot=False)
    >>> np.round(camp[:, 0], 1) #  damped natural frequencies at the first rotor speed (0 rad/s)
    array([  82.7,   86.7,  254.5,  274.3,  679.5,  716.8])
    >>> np.round(camp[:, 10], 1) # damped natural frequencies at 40 rad/s
    array([  82.6,   86.7,  254.3,  274.5,  676.5,  719.7])
    """
    rotor_state_speed = rotor.w

    speed_rad = np.array(speed_rad)
    z = []  # will contain values for each whirl (0, 0.5, 1)
    points_all = np.zeros([freqs, len(speed_rad)])

    for idx, w0, w1 in(zip(range(len(speed_rad)),
                           speed_rad[:-1],
                           speed_rad[1:])):
        # define shaft speed
        # check rotor state to avoid recalculating eigenvalues
        if not rotor.w == w0:
            rotor.w = w0

        # define x as the current speed and y as each wd
        x_w0 = np.full_like(range(freqs), w0)
        y_wd0 = rotor.wd[:freqs]

        # generate points for the first speed
        points0 = np.array([x_w0, y_wd0]).T.reshape(-1, 1, 2)
        points_all[:, idx] += y_wd0  # TODO verificar teste

        # go to the next speed
        rotor.w = w1
        x_w1 = np.full_like(range(freqs), w1)
        y_wd1 = rotor.wd[:freqs]
        points1 = np.array([x_w1, y_wd1]).T.reshape(-1, 1, 2)

        new_segment = np.concatenate([points0, points1], axis=1)

        if w0 == speed_rad[0]:
            segments = new_segment
        else:
            segments = np.concatenate([segments, new_segment])

        whirl_w = [whirl(rotor.kappa_mode(wd)) for wd in range(freqs)]
        z.append([whirl_to_cmap(i) for i in whirl_w])

    if plot is False:
        return points_all

    z = np.array(z).flatten()

    cmap = ListedColormap([c_pal['blue'],
                           c_pal['grey'],
                           c_pal['red']])

    if ax is None:
        ax = plt.gca()
    lines_2 = LineCollection(segments, array=z, cmap=cmap)
    ax.add_collection(lines_2)

    # plot harmonics in hertz
    for m in mult:
        ax.plot(speed_rad, m * speed_rad,
                color=c_pal['green2'],
                linestyle='dashed')

    # axis limits
    ax.set_xlim(0, max(speed_rad))
    ax.set_ylim(0, max(max(mult) * speed_rad))

    # legend and title
    ax.set_xlabel(r'Rotor speed ($rad/s$)')
    ax.set_ylabel(r'Damped natural frequencies ($rad/s$)')

    forward_label = mpl.lines.Line2D([], [],
                                     color=c_pal['blue'],
                                     label='Forward')
    backwardlabel = mpl.lines.Line2D([], [],
                                     color=c_pal['red'],
                                     label='Backward')
    mixedlabel = mpl.lines.Line2D([], [],
                                  color=c_pal['grey'],
                                  label='Mixed')

    ax.legend(handles=[forward_label, backwardlabel, mixedlabel],
              loc=2)

    # restore rotor speed
    rotor.w = rotor_state_speed

    return ax


# TODO critical speed map


def plot_freq_response(ax0, ax1, omega, magdb, phase, out, inp):
    art0 = ax0.plot(omega, magdb[out, inp, :])
    art1 = ax1.plot(omega, phase[out, inp, :])
    for ax in [ax0, ax1]:
        ax.set_xlim(0, max(omega))
        ax.yaxis.set_major_locator(
            mpl.ticker.MaxNLocator(prune='lower'))
        ax.yaxis.set_major_locator(
            mpl.ticker.MaxNLocator(prune='upper'))

    ax0.text(.9, .9, 'Output %s' % out,
             horizontalalignment='center',
             transform=ax0.transAxes)
    ax0.text(.9, .7, 'Input %s' % inp,
             horizontalalignment='center',
             transform=ax0.transAxes)

    ax0.set_ylabel('Magnitude $(dB)$')
    ax1.set_ylabel('Phase')
    ax1.set_xlabel('Frequency (rad/s)')

    return art0, art1


def plot_time_response(rotor, F, t, dof, ax=None):
    """Plot the time response.

    This function will take a rotor object and plot its time response
    given a force and a time.

    Parameters
    ----------
    rotor: rotor object
        A rotor object.
    F : array
        Force array (needs to have the same number of rows as time array).
        Each column corresponds to a dof and each row to a time.
    t : array
        Time array.
    dof : int
        Degree of freedom that will be observed.
    ax : matplotlib axes, optional
        Axes in which the plot will be drawn.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes object with the plot.

    Examples:
    ---------
    """
    t_, yout, xout = rotor.time_response(F, t)

    if ax is None:
        ax = plt.gca()

    ax.plot(t, yout[:, dof])

    if dof % 4 == 0:
        obs_dof = '$x$'
        amp = 'm'
    elif dof + 1 % 4 == 0:
        obs_dof = '$y$'
        amp = 'm'
    elif dof + 2 % 4 == 0:
        obs_dof = '$\alpha$'
        amp = 'rad'
    else:
        obs_dof = '$\beta$'
        amp = 'rad'

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (%s)' % amp)
    ax.set_title('Response for node %s and degree of freedom %s'
                 % (dof//4, obs_dof))

    return ax

# TODO Change results to rotor methods
# TODO Add root locus plot
# TODO Add orbit plot
# TODO Add mode shape plot
