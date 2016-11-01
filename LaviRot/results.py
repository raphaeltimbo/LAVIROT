r"""
This module contains :py:meth:`LaviRot.results` with functions
to evaluate results and functions to create plots
"""
# TODO detail the results docstring
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

c_pal = {'red': '#C93C3C',
         'blue': '#0760BA',
         'green': '#2ECC71',
         'dark blue': '#07325E',
         'purple': '#A349C6',
         'grey': '#2D2D2D',
         'green2': '#08A4AF'}

fn = os.path.join(os.path.dirname(__file__), r'styles\matplotlibrc')


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
             'bearing': '#355d7a'}

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
        ax.text(position, -0.01,
                '%.0f' % node,
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
        disk_points_l = [[zpos, -ypos],  # upper
                         [zpos + hw, -(ypos + 0.1 * D)],
                         [zpos + hw, -(ypos + 0.9 * D)],
                         [zpos - hw, -(ypos + 0.9 * D)],
                         [zpos - hw, -(ypos + 0.1 * D)],
                         [zpos, -ypos]]
        ax.add_patch(mpatches.Polygon(disk_points_u, facecolor=r_pal['disk']))
        ax.add_patch(mpatches.Polygon(disk_points_l, facecolor=r_pal['disk']))

    # plot bearings
    for bearing in rotor.bearing_elements:
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

    plt.show()
    return fig


def MAC(u, v):
    """MAC for two vectors"""
    H = lambda a: a.T.conj()
    return np.absolute((H(u) @ v)**2 / ((H(u) @ u)*(H(v) @ v)))


def MAC_modes(U, V, n=None):
    """MAC for multiple vectors"""
    # n is the number of modes to be evaluated
    if n is None:
        n = U.shape[1]
    macs = np.zeros((n, n))
    for u in enumerate(U.T[:n]):
        for v in enumerate(V.T[:n]):
            macs[u[0], v[0]] = MAC(u[1], v[1])
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


def campbell(rotor, speed_hz, freqs=6, mult=[1, 2]):
    #  TODO mult will be the harmonics for interest e.g., 1x, 2x etc.
    mpl.rcParams.update(mpl.rc_params_from_file(fn))

    z = []  # will contain values for each whirl (0, 0.5, 1)

    # input to rotor.w must be in rad/s
    # so we change from  hertz to rad/s
    speed_rad = speed_hz*2*np.pi

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

        whirl_w = [whirl(rotor.kappa_mode(wn)) for wn in range(freqs)]
        z.append([whirl_to_cmap(i) for i in whirl_w])

    z = np.array(z).flatten()

    cmap = ListedColormap([c_pal['blue'],
                           c_pal['grey'],
                           c_pal['red']])

    fig, ax = plt.subplots()
    lines_2 = LineCollection(segments, array=z, cmap=cmap)
    ax.add_collection(lines_2)

    # plot speed in hertz
    ax.plot(speed_hz, speed_hz,
             color=c_pal['green2'],
             linestyle='dashed')
    ax.plot(speed_hz, 2*speed_hz,
             color=c_pal['green2'],
             linestyle='dashed')

    # axis limits
    ax.set_xlim(0, max(speed_hz))
    ax.set_ylim(0, max(2*speed_hz))

    # legend and title
    ax.set_xlabel(r'Rotor speed ($Hz$)')
    ax.set_ylabel(r'Damped natural frequencies ($Hz$)')

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

    plt.show()
    return fig
