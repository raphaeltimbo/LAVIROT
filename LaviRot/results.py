r"""
This module contains :py:meth:`LaviRot.results` with functions
to evaluate results and functions to create plots
"""
# TODO detail the results docstring
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


__all__ = ["plot_rotor",
           "MAC",
           "MAC_modes",
           "campbell",
           "bearing_parameters"]


c_pal = {'red': '#C93C3C',
         'blue': '#0760BA',
         'green': '#2ECC71',
         'dark blue': '#07325E',
         'purple': '#A349C6',
         'grey': '#2D2D2D',
         'green2': '#08A4AF'}

fn = os.path.join(os.path.dirname(__file__), r'styles/matplotlibrc')


def plot_rotor(rotor):
    """ Plots a rotor object.

    This function will take a rotor object and plot its shaft,
    disks and bearing elements

    Parameters
    ----------
    rotor: rotor object
        A rotor object

    Returns
    -------
    Plots the rotor object.

    Examples:

    """
    mpl.rcParams.update(mpl.rc_params_from_file(fn))
    plt.rcParams['axes.facecolor'] = '#E5E5E5'
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['xtick.labelsize'] = 0
    plt.rcParams['ytick.labelsize'] = 0

    #  define a color palette for the rotor
    r_pal = {'shaft': '#525252',
             'node': '#6caed6',
             'disk': '#bc625b',
             'bearing': '#355d7a',
             'seal': '#77ACA2'}

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #  plot shaft centerline
    shaft_end = rotor.nodes_pos[-1]
    ax.plot([-.2 * shaft_end, 1.2 * shaft_end], [0, 0], 'k-.')
    max_diameter = max([disk.o_d for disk in rotor.disk_elements])
    ax.set_ylim(-1.2 * max_diameter, 1.2 * max_diameter)
    ax.axis('equal')

    #  plot nodes
    for node, position in enumerate(rotor.nodes_pos):
        ax.plot(position, 0,
                zorder=2, ls='', marker='D', color=r_pal['node'], markersize=10, alpha=0.6)
        ax.text(position, 0,
                '%.0f' % node,
                size='smaller',
                horizontalalignment='center',
                verticalalignment='center')

    # plot shaft elements
    for sh_elm in rotor.shaft_elements:
        position_u = [rotor.nodes_pos[sh_elm.n], sh_elm.i_d]  # upper
        position_l = [rotor.nodes_pos[sh_elm.n], -sh_elm.o_d + sh_elm.i_d]  # lower
        width = sh_elm.L
        height = sh_elm.o_d - sh_elm.i_d

        #  plot the upper half of the shaft
        ax.add_patch(mpatches.Rectangle(position_u, width, height,
                                        facecolor=r_pal['shaft'], alpha=0.8))
        #  plot the lower half of the shaft
        ax.add_patch(mpatches.Rectangle(position_l, width, height,
                                        facecolor=r_pal['shaft'], alpha=0.8))

    # plot disk elements
    for disk in rotor.disk_elements:
        zpos = rotor.nodes_pos[disk.n]
        ypos = rotor.shaft_elements[disk.n].o_d
        D = disk.o_d
        hw = disk.width / 2  # half width

        #  node (x pos), outer diam. (y pos)
        disk_points_u = [[zpos, ypos],  # upper
                         [zpos + hw, ypos + 0.1 * D],
                         [zpos + hw, ypos + 0.9 * D],
                         [zpos - hw, ypos + 0.9 * D],
                         [zpos - hw, ypos + 0.1 * D],
                         [zpos, ypos]]
        disk_points_l = [[zpos, -ypos],  # lower
                         [zpos + hw, -(ypos + 0.1 * D)],
                         [zpos + hw, -(ypos + 0.9 * D)],
                         [zpos - hw, -(ypos + 0.9 * D)],
                         [zpos - hw, -(ypos + 0.1 * D)],
                         [zpos, -ypos]]
        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=r_pal['disk']))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=r_pal['disk']))

    # plot bearings
    for bearing in rotor.bearing_seal_elements:
        # name is used here because classes are not import to this module
        if type(bearing).__name__ == 'BearingElement':
            zpos = rotor.nodes_pos[bearing.n]
            #  TODO this will need to be modified for tapppered elements
            #  check if the bearing is in the last node
            ypos = -rotor.nodes_o_d[bearing.n]
            h = -0.75 * ypos  # height

            #  node (x pos), outer diam. (y pos)
            bearing_points = [[zpos, ypos],  # upper
                              [zpos + h / 2, ypos - h],
                              [zpos - h / 2, ypos - h],
                              [zpos, ypos]]
            ax.add_patch(mpatches.Polygon(bearing_points, color=r_pal['bearing']))

        elif type(bearing).__name__ == 'SealElement':
            zpos = rotor.nodes_pos[bearing.n]
            #  check if the bearing is in the last node
            ypos = rotor.nodes_o_d[bearing.n]
            hw = 0.05
            # TODO adapt hw according to bal drum diameter

            #  node (x pos), outer diam. (y pos)
            seal_points_u = [[zpos, ypos*1.1],  # upper
                             [zpos + hw, ypos*1.1],
                             [zpos + hw, ypos*1.3],
                             [zpos - hw, ypos*1.3],
                             [zpos - hw, ypos*1.1],
                             [zpos, ypos*1.1]]
            seal_points_l = [[zpos, -ypos*1.1],  # lower
                             [zpos + hw, -(ypos*1.1)],
                             [zpos + hw, -(ypos*1.3)],
                             [zpos - hw, -(ypos*1.3)],
                             [zpos - hw, -(ypos*1.1)],
                             [zpos, -ypos*1.1]]
            ax.add_patch(mpatches.Polygon(seal_points_u, facecolor=r_pal['seal']))
            ax.add_patch(mpatches.Polygon(seal_points_l, facecolor=r_pal['seal']))

    return fig


def MAC(u, v):
    """MAC for two vectors"""
    H = lambda a: a.T.conj()
    return np.absolute((H(u) @ v)**2 / ((H(u) @ u)*(H(v) @ v)))


def MAC_modes(U, V, n=None, plot=True):
    """MAC for multiple vectors"""
    # n is the number of modes to be evaluated
    if n is None:
        n = U.shape[1]
    macs = np.zeros((n, n))
    for u in enumerate(U.T[:n]):
        for v in enumerate(V.T[:n]):
            macs[u[0], v[0]] = MAC(u[1], v[1])

    if not plot:
        return macs

    xpos, ypos = np.meshgrid(range(n), range(n))
    xpos, ypos = 0.5 + xpos.flatten(), 0.5 + ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = 0.75 * np.ones_like(xpos)
    dy = 0.75 * np.ones_like(xpos)
    dz = macs.T.flatten()

    fig = plt.figure(figsize=(12, 8))
    #fig.suptitle('MAC - %s vs %s' % (U.name, V.name), fontsize=12)
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=plt.cm.viridis(dz), alpha=0.7)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    #ax.set_xlabel('%s  modes' % U.name)
    #ax.set_ylabel('%s  modes' % V.name)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=0, vmax=1))
    # fake up the array of the scalar mappable
    sm._A = []
    cbar = fig.colorbar(sm, shrink=0.5, aspect=10)
    cbar.set_label('MAC')

    return macs


def whirl(kappa_mode):
    """Evaluates the whirl of a mode"""
    if all(kappa >= -1e-3 for kappa in kappa_mode):
        whirldir = 'Forward'
    elif all(kappa <= 1e-3 for kappa in kappa_mode):
        whirldir = 'Backward'
    else:
        whirldir = 'Mixed'
    return whirldir


def whirl_to_cmap(whirl):
    """Maps the whirl to a value"""
    if whirl == 'Forward':
        return 0
    elif whirl == 'Backward':
        return 1
    else:
        return 0.5


def campbell(rotor, speed_rad, freqs=6, mult=[1]):
    #  TODO docstrinc
    #  TODO mult will be the harmonics for interest e.g., 1x, 2x etc.
    mpl.rcParams.update(mpl.rc_params_from_file(fn))
    rotor_state_speed = rotor.w

    z = []  # will contain values for each whirl (0, 0.5, 1)

    for w0, w1 in (zip(speed_rad[:-1], speed_rad[1:])):
        # define shaft speed
        # check rotor state to avoid recalculating eigenvalues
        if not rotor.w == w0:
            rotor.w = w0

        # define x as the current speed and y as each wd
        x_w0 = np.full_like(range(freqs), w0)
        y_wd0 = rotor.wd[:freqs]

        # generate points for the first speed
        points0 = np.array([x_w0, y_wd0]).T.reshape(-1, 1, 2)

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

    z = np.array(z).flatten()

    cmap = ListedColormap([c_pal['blue'],
                           c_pal['grey'],
                           c_pal['red']])

    fig, ax = plt.subplots()
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

    return fig


def bearing_parameters(bearing):
    mpl.rcParams.update(mpl.rc_params_from_file(fn))
    fig, ax = plt.subplots(2, sharex=True)

    w = np.linspace(0, 1.3*bearing.w[-1], 1000)

    for a in ax:
        a.ticklabel_format(style='sci',
                           axis='both',
                           scilimits=(0, 0))

    ax[0].set_ylabel(r'Bearing Stiffness ($N/m$)')
    ax[1].set_ylabel(r'Bearing Damping ($Ns/m$)')
    ax[1].set_xlabel(r'Speed (rad/s)')

    ax[0].plot(w, bearing.kxx(w))
    ax[0].plot(w, bearing.kyy(w))
    ax[0].plot(w, bearing.kxy(w))
    ax[0].plot(w, bearing.kyx(w))

    ax[1].plot(w, bearing.cxx(w))
    ax[1].plot(w, bearing.cyy(w))
    ax[1].plot(w, bearing.cxy(w))
    ax[1].plot(w, bearing.cyx(w))

    return fig


# TODO critical speed map
