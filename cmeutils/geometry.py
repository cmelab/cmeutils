import math

import numpy as np
from numpy.linalg import svd


def get_plane_normal(points):
    """Calculate the plane which best fits a cloud of points.

    Best fit is calculated using numpy's Singular Value Decomposition.

    Example
    -------
    To visualize the plane fit in a Jupyter notebook::

        %matplotlib notebook
        import matplotlib.pyplot as plt
        import numpy as np

        normal, ctr = get_plane_normal(points)
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

        # plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 3), np.linspace(ylim[0], ylim[1], 3)
        )

        d = -ctr.dot(normal)
        ax.scatter(ctr[0], ctr[1], ctr[2], color='r')
        z = (-normal[0] * xx - normal[1] * yy - d) * 1 / normal[2]
        ax.plot_surface(xx, yy, z, alpha=0.5)

        ax.set_box_aspect(aspect = (1,1,1))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


    Parameters
    ----------
    points : numpy.ndarray, shape (N,3)
        Coordinates (x,y,z) through which to fit a plane. Must be at least 3
        points.

    Returns
    -------
    ctr : numpy.ndarray, shape (3,)
        The center of the point cloud.
    normal : numpy.ndarray, shape (3,)
        The plane normal vector. A unit vector from the origin.
    """
    assert points.shape[0] >= 3, "Need at least 3 points to calculate a plane."
    ctr = points.mean(axis=0)
    shiftpoints = points - ctr
    U, S, Vt = svd(shiftpoints.T @ shiftpoints)
    normal = U[:, -1]
    return ctr, normal


def angle_between_vectors(u, v, min_angle=True, degrees=True):
    """Calculate the angle between two vectors in radians or degrees.

    Parameters
    ----------
    u : np.ndarray, shape (3,)
        Vector
    v : np.ndarray, shape (3,)
        Vector
    min_angle : bool, default True
        Whether to return the supplement if the angle is greater than 90
        degrees. Useful for calculating the minimum angle between the normal
        vectors of planes as direction doesn't matter.
    degrees : bool, default True
        If True, the angle values are returned in degrees.
        If False, the angle values are returned in radians.

    Returns
    -------
    angle: float
        Angle between u and v

    """
    # Angle in radians
    angle = np.arccos(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    if angle > math.pi/2 and min_angle:
        angle = math.pi - angle

    if degrees:
        return np.rad2deg(angle)
    else:
        return angle
