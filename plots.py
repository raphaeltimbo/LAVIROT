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

    #  plot nodes
    for node, position in enumerate(rotor.nodes_pos):
        ax.plot(position, 0,
                zorder=2, ls='', marker='D', color="#413cd6", markersize=10, alpha=0.6)
        ax.text(position -0.004, -0.01,
                '%.0f' % node)

    #  plot shaft elements
    for sh_elm in rotor.shaft_elements:
        up_position = [sh_elm.z, sh_elm.i_d]
        down_position = [sh_elm.z, -sh_elm.o_d + sh_elm.i_d]
        width = sh_elm.L
        height = sh_elm.o_d - sh_elm.i_d

        #  plot the upper half of the shaft
        ax.add_patch(mpatches.Rectangle(up_position, width, height,
                                        facecolor="#aeaeae", edgecolor='#767676', alpha=0.8))
        #  plot the lower half of the shaft
        ax.add_patch(mpatches.Rectangle(down_position, width, height,
                                        facecolor="#aeaeae", edgecolor='#767676', alpha=0.8))

    plt.show()