r"""
This module contains :py:meth:`LaviRot.results` with functions
to evaluate results and functions to create plots
"""
# TODO detail the results docstring
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('ggplot')
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

    #  plot nodes
    for node, position in enumerate(rotor.nodes_pos):
        ax.plot(position, 0,
                zorder=2, ls='', marker='D', color=r_pal['node'], markersize=10, alpha=0.6)
        ax.text(position - 0.02, -0.008,
                '%.0f' % node)

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
        h = -0.5 * ypos  # height

        #  node (x pos), outer diam. (y pos)
        bearing_points = [[zpos, ypos],  # upper
                          [zpos + h / 2, ypos - h],
                          [zpos - h / 2, ypos - h],
                          [zpos, ypos]]
        ax.add_patch(mpatches.Polygon(bearing_points, color=r_pal['bearing']))

    plt.show()


def MAC(u, v):
    H = lambda a: a.T.conj()
    return np.absolute((H(u) @ v)**2 / ((H(u) @ u)*(H(v) @ v)))


def MAC_modes(U, V, n=None):
    # n is the number of modes to be evaluated
    if n is None:
        n = U.shape[1]
    macs = np.zeros((n, n))
    for u in enumerate(U.T[:n]):
        for v in enumerate(V.T[:n]):
            macs[u[0], v[0]] = MAC(u[1], v[1])
    return macs
