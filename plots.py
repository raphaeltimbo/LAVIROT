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
    ----------
    Plots the rotor object.

    Examples:

    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['xtick.labelsize'] = 0
    plt.rcParams['ytick.labelsize'] = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #  plot shaft centerline
    shaft_end = rotor.nodes_pos[-1]
    ax.plot([-.2 * shaft_end, 1.2 * shaft_end], [0, 0], 'k-.')
    max_diameter = max([disk.o_d for disk in rotor.disk_elements])
    ax.set_ylim(-1.2*max_diameter, 1.2*max_diameter)

    #  plot nodes
    for node, position in enumerate(rotor.nodes_pos):
        ax.plot(position, 0,
                zorder=2, ls='', marker='D', color="#413cd6", markersize=10, alpha=0.6)
        ax.text(position -0.004, -0.01,
                '%.0f' % node)

    #  plot shaft elements
    for sh_elm in rotor.shaft_elements:
        position_u = [sh_elm.z, sh_elm.i_d]    # upper
        position_l = [sh_elm.z, -sh_elm.o_d + sh_elm.i_d]    # lower
        width = sh_elm.L
        height = sh_elm.o_d - sh_elm.i_d

        #  plot the upper half of the shaft
        ax.add_patch(mpatches.Rectangle(position_u, width, height,
                                        facecolor="#aeaeae", edgecolor='#767676', alpha=0.8))
        #  plot the lower half of the shaft
        ax.add_patch(mpatches.Rectangle(position_l, width, height,
                                        facecolor="#aeaeae", edgecolor='#767676', alpha=0.8))

    #  plot disk elements
    for disk in rotor.disk_elements:
        zpos = rotor.nodes_pos[disk.n]
        ypos = rotor.shaft_elements[disk.n].o_d
        D = disk.o_d
        hw = disk.width/2    # half width

        #  node (x pos), outer diam. (y pos)
        disk_points_u = [[zpos, ypos],    # upper
                         [zpos + hw, ypos + 0.1*D],
                         [zpos + hw, ypos + 0.9*D],
                         [zpos - hw, ypos + 0.9*D],
                         [zpos - hw, ypos + 0.1*D],
                         [zpos, ypos]]
        disk_points_l = [[zpos, -(ypos)],    # upper
                         [zpos + hw, -(ypos + 0.1*D)],
                         [zpos + hw, -(ypos + 0.9*D)],
                         [zpos - hw, -(ypos + 0.9*D)],
                         [zpos - hw, -(ypos + 0.1*D)],
                         [zpos, -(ypos)]]
        ax.add_patch(mpatches.Polygon(disk_points_u))
        ax.add_patch(mpatches.Polygon(disk_points_l))

    #  plot bearings
    for bearing in rotor.bearing_elements:
        zpos = rotor.nodes_pos[bearing.n]
        #  TODO this will need to be modified for tapppered elements
        #  check if the bearing is in the last node
        ypos = -rotor.nodes_o_d[bearing.n]
        h = -0.5*ypos   # height

        #  node (x pos), outer diam. (y pos)
        bearing_points = [[zpos, ypos],    # upper
                         [zpos + h/2, ypos - h],
                         [zpos - h/2, ypos - h],
                         [zpos, ypos]]
        ax.add_patch(mpatches.Polygon(bearing_points, color='#e75555'))

    plt.show()