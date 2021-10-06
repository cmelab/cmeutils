from numpy.linalg import svd


def get_plane_normal(points):
    """Calculate the plane which best fits a cloud of points.

    Best fit is calculated using numpy's Singular Value Decomposition.

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


