"""
This modules provides functions to load xyz files
    and claculate the radial distribution function
"""
import re
import numpy as np
from itertools import product
from scipy.spatial.distance import cdist


def get_frames_from_xyz(filename, ncols=3):
    """
    Load the configurations of particles inside an xyz file

    Args:
        filename (str): the name/path of xyz file
        ncols (int): the number of columns to read, 3 = x, y, z

    Return:
        np.ndarray: the positions of particles in each frame,
            shape (n_frame, n_particle, n_cols)
    """
    f = open(filename, 'r')
    frames = []
    for line in f:
        is_head = re.match(r'(\d+)\n', line)
        if is_head:
            frames.append([])
            particle_num = int(is_head.group(1))
            f.readline()  # jump through comment line
            for j in range(particle_num):
                data = re.split(r'\s', f.readline())[1: 1 + ncols]
                frames[-1].append(list(map(float, data)))
    f.close()
    return np.array(frames)


def get_rdf(positions, box, cutoff, bins):
    """
    Calculate the radial distribution function (g or r) given particles
        inside a box with periodic boundary

    Args:
        positiosn (np.ndarray): the positions of all particles, shape (n, dim)
        box (iterable or float): the length of the box with periodic boundaries.
            if given a number, then assume the box is inside a cubic box
            if given a iterable, its length must be the dimension
        cutoff (float): the maximum r value in the rdf
        bins (iterable or int): specifying the bins of the radius
            if given a number, the bin edges are `np.linspace(0, cutoff, bins + 1)`
            if given an iterable, then the bin edges equals `bins`

    Return:
        tuple: the rdial distribution function in the form of (r, g(r))

    Example:
        >>> np.random.seed(0)
        >>> ideal_gas = np.random.random((250, 2))
        >>> r, rdf = get_rdf(ideal_gas, box=1, cutoff=3, bins=20)
        >>> np.isclose(rdf, 1, atol=0.1).all()
        True
        >>> len(r), len(rdf)
        (20, 20)
    """
    # processing parameters
    n, dim = positions.shape
    if type(box) in [int, float]:
        box = np.array([box] * dim, dtype=float)
    else:
        box = np.fromiter(box, dtype=float)
    if len(box) != dim:
        raise ValueError(
        "".join(["The size of the box (", str(len(box)),
        ") does not match the dimension (", str(dim), ")"])
    )
    if type(bins) == int:
        bins = np.linspace(0, cutoff, bins + 1)
    else:
        bins = np.fromtier(bins, dtype=float)
    # calculate the density
    box_volumn = 1
    for box_1d in box:
        box_volumn *= box_1d
    density = n / box_volumn

    n_level = np.ceil(cutoff / box).astype(int).max()  # n = neighbour
    n_indices = np.arange(-n_level, n_level + 1)  # e.g. (-2, -1, 0, 1, 2)
    n_indices = list(product(n_indices, repeat=dim))
    n_shift = np.array(n_indices) * box[None, :]
    n_cells = n_shift[:, None, :] + positions[None, :, :] # shape (n_cell, n, dim)
    distances = [cdist(positions, nc) for nc in n_cells]
    for i in range(n_cells.shape[0]):  # remove self overlapping
        np.fill_diagonal(distances[i], np.nan)
    hist, _ = np.histogram(np.concatenate(distances).ravel(), bins=bins)

    r = (bins[1:] + bins[:-1]) / 2
    dr = bins[1] - bins[0]
    if dim == 1:
        v_shell = 2 * dr
    elif dim == 2:
        v_shell = 2 * np.pi * r * dr
    elif dim == 3:
        v_shell = 4 * np.pi * r**2 * dr
    else:
        raise ValueError(r"RDF for dimensions higher than 3 not implemented")
    rdf = hist / (n * v_shell  * density)
    return r, rdf


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 14
    import matplotlib.pyplot as plt

    frames = get_frames_from_xyz('positions_hard_sphere.xyz')
    cutoff = 10
    box = 18.70550963409122

    rdfs = []
    for xyz in frames:
        r, rdf = get_rdf(xyz, box=box, cutoff=cutoff, bins=50)
        rdfs.append(rdf)

    plt.plot(r, np.mean(rdfs, axis=0), color='teal')
    plt.plot((0, cutoff), (1, 1), lw=1, color='k')
    plt.xlim(0, cutoff)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.tight_layout()
    plt.savefig('rdf_hard_sphere.pdf')
    plt.show()
