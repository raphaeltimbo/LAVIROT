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
