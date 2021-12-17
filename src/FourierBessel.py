import numpy as np
from numpy import diff, exp, log, pi
from scipy.special import jv
from scipy.sparse.linalg import LinearOperator, cg
import faiss
import time
from tqdm import tqdm
"""
Modified version of the ASPIR toolbox: https://github.com/ComputationalCryoEM/ASPIRE-Python/tree/3223d6c2c308af66f321ca2b0ddf3933045f904b
"""

def ensure(cond, error_message=None):
    """
    assert statements in Python are sometimes optimized away by the compiler, and are for internal testing purposes.
    For user-facing assertions, we use this simple wrapper to ensure conditions are met at relevant parts of the code.
    :param cond: Condition to be ensured
    :param error_message: An optional error message if condition is not met
    :return: If condition is met, returns nothing, otherwise raises AssertionError
    """
    if not cond:
        raise AssertionError(error_message)
  

def m_reshape(x, new_shape):
    # This is a somewhat round-about way of saying:
    #   return x.reshape(new_shape, order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    # Note that flattening is required before reshaping, because
    if isinstance(new_shape, tuple):
        return m_flatten(x).reshape(new_shape[::-1]).T
    else:
        return x

def m_flatten(x):
    # This is a somewhat round-about way of saying:
    #   return x.flatten(order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    return x.T.flatten()

def check_besselj_zeros(nu, z):
    dz = np.diff(z)
    ddz = np.diff(dz)

    result = True
    result = result and all(np.isreal(z))
    result = result and z[0] > 0
    result = result and all(dz > 3)

    if nu >= 0.5:
        result = result and all(ddz < 16 * np.spacing(z[1:-1]))
    else:
        result = result and all(ddz > -16 * np.spacing(z[1:-1]))

    return result
   
def besselj_newton(nu, z0, max_iter=10):
    z = z0

    # Factor worse than machine precision
    c = 8

    for i in range(max_iter):
        # Calculate values and derivatives at z
        f = jv(nu, z)
        fp = jv(nu - 1, z) - nu * f / z

        # Update zeros
        dz = -f / fp
        z = z + dz

        # Check for convergence
        if all(np.abs(dz) < c * np.spacing(z)):
            break

        # If we're not converging yet, start relaxing convergence criterion
        if i >= 6:
            c *= 2

    return z

def besselj_zeros(nu, k):
    ensure(k >= 3, "k must be >= 3")
    ensure(0 <= nu <= 1e7, "nu must be between 0 and 1e7")

    z = np.zeros(k)

    # Guess first zeros using powers of nu
    c0 = np.array(
        [
            [0.1701, -0.6563, 1.0355, 1.8558],
            [0.1608, -1.0189, 3.1348, 3.2447],
            [-0.2005, -1.2542, 5.7249, 4.3817],
        ]
    )
    z0 = nu + c0 @ ((nu + 1) ** np.array([[-1, -2 / 3, -1 / 3, 1 / 3]]).T)

    # refine guesses
    z[:3] = besselj_newton(nu, z0).squeeze()

    n = 3
    j = 2
    err_tol = 5e-3

    # Estimate further zeros iteratively using spacing of last three zeros so far
    while n < k:
        j = min(j, k - n)

        # Use last 3 zeros to predict spacing for next j zeros
        r = diff(z[n - 3 : n]) - pi
        if (r[0] * r[1]) > 0 and (r[0] / r[1]) > 1:
            p = log(r[0] / r[1]) / log(1 - 1 / (n - 1))
            t = np.array(np.arange(1, j + 1), ndmin=2).T / (n - 1)
            dz = pi + r[1] * exp(p * log(1 + t))
        else:
            dz = pi * np.ones((j, 1))

        # Guess and refine
        z0 = z[n - 1] + np.cumsum(dz)
        z[n : n + j] = besselj_newton(nu, z0)

        # Check to see that the sequence of zeros makes sense
        ensure(
            check_besselj_zeros(nu, z[n - 2 : n + j]),
            "Unable to properly estimate Bessel function zeros.",
        )

        # Check how far off we are
        err = (z[n : n + j] - z0) / np.diff(z[n - 1 : n + j])

        n = n + j
        if max(abs(err)) < err_tol:
            # Predictions were close enough, double number of zeros
            j *= 2
        else:
            # Some predictions were off, set to double the number of good predictions
            j = 2 * (np.where(abs(err) >= err_tol)[0][0] + 1)

    return z

def num_besselj_zeros(ell, r):
    k = 4
    r0 = besselj_zeros(ell, k)
    while all(r0 < r):
        k *= 2
        r0 = besselj_zeros(ell, k)
    r0 = r0[r0 < r]
    return len(r0), r0


def cart2pol(x, y):
    """
    Convert Cartesian to Polar Coordinates. All input arguments must be the same shape.
    :param x: x-coordinate in Cartesian space
    :param y: y-coordinate in Cartesian space
    :return: A 2-tuple of values:
        theta: angular coordinate/azimuth
        r: radial distance from origin
    """
    return np.arctan2(y, x), np.hypot(x, y)


def cart2sph(x, y, z):
    """
    Transform cartesian coordinates to spherical. All input arguments must be the same shape.
    :param x: X-values of input co-ordinates.
    :param y: Y-values of input co-ordinates.
    :param z: Z-values of input co-ordinates.
    :return: A 3-tuple of values, all of the same shape as the inputs.
        (<azimuth>, <elevation>, <radius>)
    azimuth and elevation are returned in radians.
    This function is equivalent to MATLAB's cart2sph function.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def grid_2d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate two dimensional grid.
    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and polar coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n / 2, n / 2, dtype=dtype))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2, dtype=dtype)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x, y = np.meshgrid(grid, grid, indexing="ij")
    phi, r = cart2pol(x, y)

    return {"x": x, "y": y, "phi": phi, "r": r}


def grid_3d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate three dimensional grid.
    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and spherical coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n / 2, n / 2, dtype=dtype))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2, dtype=dtype)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}

def unique_coords_nd(N, ndim, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate unique polar coordinates from 2D or 3D rectangular coordinates.
    :param N: length size of a square or cube.
    :param ndim: number of dimension, 2 or 3.
    :param shifted: shifted half pixel or not for odd N.
    :param normalized: normalize the grid or not.
    :return: The unique polar coordinates in 2D or 3D
    """
    ensure(
        ndim in (2, 3), "Only two- or three-dimensional basis functions are supported."
    )
    ensure(N > 0, "Number of grid points should be greater than 0.")

    if ndim == 2:
        grid = grid_2d(N, shifted=shifted, normalized=normalized, dtype=dtype)
        mask = grid["r"] <= 1

        # Minor differences in r/theta/phi values are unimportant for the purpose
        # of this function, so round off before proceeding

        # TODO: numpy boolean indexing will return a 1d array (like MATLAB)
        # However, it always searches in row-major order, unlike MATLAB (column-major),
        # with no options to change the search order. The results we'll be getting back are thus not comparable.
        # We transpose the appropriate ndarrays before applying the mask to obtain the same behavior as MATLAB.
        r = grid["r"].T[mask].round(5)
        phi = grid["phi"].T[mask].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(phi, return_inverse=True)

    else:
        grid = grid_3d(N, shifted=shifted, normalized=normalized, dtype=dtype)
        mask = grid["r"] <= 1

        # In Numpy, elements in the indexed array are always iterated and returned in row-major (C-style) order.
        # To emulate a behavior where iteration happens in Fortran order, we swap axes 0 and 2 of both the array
        # being indexed (r/theta/phi), as well as the mask itself.
        # TODO: This is only for the purpose of getting the same behavior as MATLAB while porting the code, and is
        # likely not needed in the final version.

        # Minor differences in r/theta/phi values are unimportant for the purpose of this function,
        # so we round off before proceeding.

        mask_ = np.swapaxes(mask, 0, 2)
        r = np.swapaxes(grid["r"], 0, 2)[mask_].round(5)
        theta = np.swapaxes(grid["theta"], 0, 2)[mask_].round(5)
        phi = np.swapaxes(grid["phi"], 0, 2)[mask_].round(5)

        r_unique, r_idx = np.unique(r, return_inverse=True)
        ang_unique, ang_idx = np.unique(
            np.vstack([theta, phi]), axis=1, return_inverse=True
        )

    return {
        "r_unique": r_unique,
        "ang_unique": ang_unique,
        "r_idx": r_idx,
        "ang_idx": ang_idx,
        "mask": mask,
    }

def unroll_dim(X, dim):
    # TODO: dim is still 1-indexed like in MATLAB to reduce headaches for now
    # TODO: unroll/roll are great candidates for a context manager since they're always used in conjunction.
    dim = dim - 1
    old_shape = X.shape
    new_shape = old_shape[:dim]

    new_shape += (-1,)

    Y = m_reshape(X, new_shape)

    removed_dims = old_shape[dim:]

    return Y, removed_dims


def roll_dim(X, dim):
    # TODO: dim is still 1-indexed like in MATLAB to reduce headaches for now
    old_shape = X.shape
    new_shape = old_shape[:-1] + dim
    Y = m_reshape(X, new_shape)
    return Y

def complex_type(realtype):
    """
    Get Numpy complex type from corresponding real type
    :param realtype: Numpy real type
    :return complextype: Numpy complex type
    """
    realtype = np.dtype(realtype)
    complextype = None
    if realtype == np.float32:
        complextype = np.complex64
    elif realtype == np.float64:
        complextype = np.complex128
    elif realtype in (np.complex64, np.complex128):
        # logger.debug(f"Corresponding type is already complex {realtype}.")
        complextype = realtype
    else:
        msg = f"Corresponding complex type is not defined for {realtype}."
        # logger.error(msg)
        raise TypeError(msg)

    return complextype

def real_type(complextype):
    """
    Get Numpy real type from corresponding complex type
    :param complextype: Numpy complex type
    :return realtype: Numpy real type
    """
    complextype = np.dtype(complextype)
    realtype = None
    if complextype == np.complex64:
        realtype = np.float32
    elif complextype == np.complex128:
        realtype = np.float64
    elif complextype in (np.float32, np.float64):
        # logger.debug(f"Corresponding type is already real {complextype}.")
        realtype = complextype
    else:
        msg = f"Corresponding real type is not defined for {complextype}."
        # logger.error(msg)
        raise TypeError(msg)

    return realtype

class Basis:
    """
    Define a base class for expanding 2D particle images and 3D structure volumes
    """

    def __init__(self, size, ell_max=None, dtype=np.float32):
        """
        Initialize an object for the base of basis class
        :param size: The size of the vectors for which to define the basis.
            Currently only square images and cubic volumes are supported.
        :ell_max: The maximum order ell of the basis elements. If no input
            (= None), it will be set to np.Inf and the basis includes all
            ell such that the resulting basis vectors are concentrated
            below the Nyquist frequency (default Inf).
        """
        if ell_max is None:
            ell_max = np.inf

        ndim = len(size)
        nres = size[0]
        self.sz = size
        self.nres = nres
        self.count = 0
        self.ell_max = ell_max
        self.ndim = ndim
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise NotImplementedError(
                "Currently only implemented for float32 and float64 types"
            )

        self._build()

    def _getfbzeros(self):
        """
        Generate zeros of Bessel functions
        """
        # get upper_bound of zeros of Bessel functions
        upper_bound = min(self.ell_max + 1, 2 * self.nres + 1)

        # List of number of zeros
        n = []
        # List of zero values (each entry is an ndarray; all of possibly different lengths)
        zeros = []

        # generate zeros of Bessel functions for each ell
        for ell in range(upper_bound):
            _n, _zeros = num_besselj_zeros(
                ell + (self.ndim - 2) / 2, self.nres * np.pi / 2
            )
            if _n == 0:
                break
            else:
                n.append(_n)
                zeros.append(_zeros)

        #  get maximum number of ell
        self.ell_max = len(n) - 1

        #  set the maximum of k for each ell
        self.k_max = np.array(n, dtype=int)

        max_num_zeros = max(len(z) for z in zeros)
        for i, z in enumerate(zeros):
            zeros[i] = np.hstack(
                (z, np.zeros(max_num_zeros - len(z), dtype=self.dtype))
            )

        self.r0 = m_reshape(np.hstack(zeros), (-1, self.ell_max + 1)).astype(self.dtype)

    def _build(self):
        """
        Build the internal data structure to represent basis
        """
        raise NotImplementedError("subclasses must implement this")

    def indices(self):
        """
        Create the indices for each basis function
        """
        raise NotImplementedError("subclasses must implement this")

    def _precomp(self):
        """
        Precompute the basis functions at defined sample points
        """
        raise NotImplementedError("subclasses must implement this")

    def norms(self):
        """
        Calculate the normalized factors of basis functions
        """
        raise NotImplementedError("subclasses must implement this")

    def evaluate(self, v):
        """
        Evaluate coefficient vector in basis
        :param v: A coefficient vector (or an array of coefficient vectors)
            to be evaluated. The first dimension must equal `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose first dimensions equal `self.z` and the
            remaining dimensions correspond to dimensions two and higher of `v`.
        """
        raise NotImplementedError("subclasses must implement this")

    def evaluate_t(self, v):
        """
        Evaluate coefficient in dual basis
        :param v: The coefficient array to be evaluated. The first dimensions
            must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual
            basis of `basis`.
            This is an array of vectors whose first dimension equals `self.count`
            and whose remaining dimensions correspond to higher dimensions of `v`.
        """
        raise NotImplementedError("Subclasses should implement this")

    # def mat_evaluate(self, V):
    #     """
    #     Evaluate coefficient matrix in basis
    #     :param V: A coefficient matrix of size `self.count`-by-
    #         `self.count` to be evaluated.
    #     :return: A multidimensional matrix of size `self.sz`-by
    #         -`self.sz` corresponding to the evaluation of `V` in
    #         this basis.
    #     """
    #     return mdim_mat_fun_conj(V, 1, len(self.sz), self.evaluate)

    # def mat_evaluate_t(self, X):
    #     """
    #     Evaluate coefficient matrix in dual basis
    #     :param X: The coefficient array of size `self.sz`-by-`self.sz`
    #         to be evaluated.
    #     :return: The evaluation of `X` in the dual basis. This is
    #         `self.count`-by-`self.count`. matrix.
    #         If `V` is a matrix of size `self.count`-by-`self.count`,
    #         `B` is the change-of-basis matrix of `basis`, and `x` is a
    #         multidimensional matrix of size `basis.sz`-by-`basis.sz`, the
    #         function calculates V = B' * X * B, where the rows of `B`, rows
    #         of 'X', and columns of `X` are read as vectorized arrays.
    #     """
    #     return mdim_mat_fun_conj(X, len(self.sz), 1, self.evaluate_t)

    def expand(self, x):
        """
        Obtain coefficients in the basis from those in standard coordinate basis
        This is a similar function to evaluate_t but with more accuracy by using
        the cg optimizing of linear equation, Ax=b.
        :param x: An array whose last two or three dimensions are to be expanded
            the desired basis. These dimensions must equal `self.sz`.
        :return : The coefficients of `v` expanded in the desired basis.
            The last dimension of `v` is with size of `count` and the
            first dimensions of the return value correspond to
            those first dimensions of `x`.
        """
        # ensure the first dimensions with size of self.sz
        sz_roll = x.shape[: -self.ndim]

        x = x.reshape((-1, *self.sz))

        ensure(
            x.shape[-self.ndim :] == self.sz,
            f"Last {self.ndim} dimensions of x must match {self.sz}.",
        )

        operator = LinearOperator(
            shape=(self.count, self.count),
            matvec=lambda v: self.evaluate_t(self.evaluate(v)),
            dtype=self.dtype,
        )

        # TODO: (from MATLAB implementation) - Check that this tolerance make sense for multiple columns in v
        tol = 10 * np.finfo(x.dtype).eps
        # logger.info("Expanding array in basis")

        # number of image samples
        n_data = x.shape[0]
        v = np.zeros((n_data, self.count), dtype=x.dtype)

        for isample in range(0, n_data):
            b = self.evaluate_t(x[isample]).T
            # TODO: need check the initial condition x0 can improve the results or not.
            v[isample], info = cg(operator, b, tol=tol, atol=0)
            if info != 0:
                raise RuntimeError("Unable to converge!")

        # return v coefficients with the last dimension of self.count
        v = v.reshape((-1, *sz_roll))
        return v


class FBBasis2D(Basis):
    """
    Define a derived class using the Fourier-Bessel basis for mapping 2D images
    The expansion coefficients of 2D images on this basis are obtained by
    the least squares method. The algorithm is described in the publication:
    Z. Zhao, A. Singer, Fourier-Bessel Rotational Invariant Eigenimages,
    The Journal of the Optical Society of America A, 30 (5), pp. 871-877 (2013).
    """

    # TODO: Methods that return dictionaries should return useful objects instead
    def __init__(self, size, ell_max=None, dtype=np.float32):
        """
        Initialize an object for the 2D Fourier-Bessel basis class
        :param size: The size of the vectors for which to define the basis.
            Currently only square images are supported.
        :ell_max: The maximum order ell of the basis elements. If no input
            (= None), it will be set to np.Inf and the basis includes all
            ell such that the resulting basis vectors are concentrated
            below the Nyquist frequency (default Inf).
        """

        # ndim = len(size)
        # ensure(ndim == 2, "Only two-dimensional basis functions are supported.")
        # ensure(len(set(size)) == 1, "Only square domains are supported.")
        super().__init__(size, ell_max, dtype=dtype)

    def _build(self):
        """
        Build the internal data structure to 2D Fourier-Bessel basis
        """
        # logger.info(
        #     "Expanding 2D images in a spatial-domain Fourier–Bessel"
        #     " basis using the direct method."
        # )

        # get upper bound of zeros, ells, and ks  of Bessel functions
        self._getfbzeros()

        # calculate total number of basis functions
        self.count = self.k_max[0] + sum(2 * self.k_max[1:])

        # obtain a 2D grid to represent basis functions
        self.basis_coords = unique_coords_nd(self.nres, self.ndim, dtype=self.dtype)

        # generate 1D indices for basis functions
        self._compute_indices()
        self._indices = self.indices()

        # get normalized factors
        self.radial_norms, self.angular_norms = self.norms()

        # precompute the basis functions in 2D grids
        self._precomp = self._precomp()

    def _compute_indices(self):
        """
        Create the indices for each basis function
        """
        indices_ells = np.zeros(self.count, dtype=int)
        indices_ks = np.zeros(self.count, dtype=int)
        indices_sgns = np.zeros(self.count, dtype=int)

        # We'll also generate a mapping for complex construction
        self.complex_count = sum(self.k_max)
        # These map indices in complex array to pair of indices in real array
        self._pos = np.zeros(self.complex_count, dtype=int)
        self._neg = np.zeros(self.complex_count, dtype=int)

        i = 0
        ci = 0
        for ell in range(self.ell_max + 1):
            sgns = (1,) if ell == 0 else (1, -1)
            ks = np.arange(0, self.k_max[ell])

            for sgn in sgns:
                rng = np.arange(i, i + len(ks))
                indices_ells[rng] = ell
                indices_ks[rng] = ks
                indices_sgns[rng] = sgn

                if sgn == 1:
                    self._pos[ci + ks] = rng
                elif sgn == -1:
                    self._neg[ci + ks] = rng

                i += len(ks)

            ci += len(ks)

        self.angular_indices = indices_ells
        self.radial_indices = indices_ks
        self.signs_indices = indices_sgns
        # Relating to paper: a[i] = a_ell_ks = a_angularindices[i]_radialindices[i]
        self.complex_angular_indices = indices_ells[self._pos]  # k
        self.complex_radial_indices = indices_ks[self._pos]  # q

    def indices(self):
        """
        Return the precomputed indices for each basis function.
        """
        return {
            "ells": self.angular_indices,
            "ks": self.radial_indices,
            "sgns": self.signs_indices,
        }

    def _precomp(self):
        """
        Precompute the basis functions at defined sample points
        """

        r_unique = self.basis_coords["r_unique"]
        ang_unique = self.basis_coords["ang_unique"]

        ind_radial = 0
        ind_ang = 0

        radial = np.zeros(shape=(len(r_unique), np.sum(self.k_max)), dtype=self.dtype)
        ang = np.zeros(
            shape=(ang_unique.shape[-1], 2 * self.ell_max + 1), dtype=self.dtype
        )

        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                # Only normalized by the radial part of basis function
                radial[:, ind_radial] = (
                    jv(ell, self.r0[k - 1, ell] * r_unique)
                    / self.radial_norms[ind_radial]
                )
                ind_radial += 1

            sgns = (1,) if ell == 0 else (1, -1)
            for sgn in sgns:
                fn = np.cos if sgn == 1 else np.sin
                ang[:, ind_ang] = fn(ell * ang_unique)
                ind_ang += 1

        return {"radial": radial, "ang": ang}

    def norms(self):
        """
        Calculate the normalized factors of basis functions
        """
        radial_norms = np.zeros(np.sum(self.k_max), dtype=self.dtype)
        angular_norms = np.zeros(np.sum(self.k_max), dtype=self.dtype)
        norm_fn = self.basis_norm_2d

        i = 0
        for ell in range(0, self.ell_max + 1):
            for k in range(1, self.k_max[ell] + 1):
                radial_norms[i], angular_norms[i] = norm_fn(ell, k)
                i += 1

        return radial_norms, angular_norms

    def basis_norm_2d(self, ell, k):
        """
        Calculate the normalized factors from radial and angular parts of a specified basis function
        """
        rad_norm = (
            np.abs(jv(ell + 1, self.r0[k - 1, ell]))
            * np.sqrt(1 / 2.0)
            * self.nres
            / 2.0
        )
        ang_norm = np.sqrt(np.pi)
        if ell == 0:
            ang_norm *= np.sqrt(2)

        return rad_norm, ang_norm

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in FB basis
        :param v: A coefficient vector (or an array of coefficient vectors) to
            be evaluated. The last dimension must equal `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose last dimensions equal `self.sz` and the remaining
            dimensions correspond to first dimensions of `v`.
        """

        # if v.dtype != self.dtype:
        #     logger.warning(
        #         f"{self.__class__.__name__}::evaluate"
        #         f" Inconsistent dtypes v: {v.dtype} self: {self.dtype}"
        #     )

        # Transpose here once, instead of several times below  #RCOPT
        v = v.reshape(-1, self.count).T

        r_idx = self.basis_coords["r_idx"]
        ang_idx = self.basis_coords["ang_idx"]
        mask = m_flatten(self.basis_coords["mask"])

        ind = 0
        ind_radial = 0
        ind_ang = 0

        x = np.zeros(shape=tuple([np.prod(self.sz)] + list(v.shape[1:])), dtype=v.dtype)
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            idx_radial = ind_radial + np.arange(0, k_max, dtype=int)

            # include the normalization factor of angular part
            ang_nrms = self.angular_norms[idx_radial]
            radial = self._precomp["radial"][:, idx_radial]
            radial = radial / ang_nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp["ang"][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max, dtype=int)
                x[mask] += ang_radial @ v[idx]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        x = x.T.reshape(-1, *self.sz)  # RCOPT

        return x

    def evaluate_t(self, v):
        """
        Evaluate coefficient in FB basis from those in standard 2D coordinate basis
        :param v: The coefficient array to be evaluated. The last dimensions
            must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual basis
            of `basis`. This is an array of vectors whose last dimension equals
             `self.count` and whose first dimensions correspond to
             first dimensions of `v`.
        """

        # if v.dtype != self.dtype:
        #     logger.warning(
        #         f"{self.__class__.__name__}::evaluate_t"
        #         f" Inconsistent dtypes v: {v.dtype} self: {self.dtype}"
        #     )

        # if isinstance(v, Image):
        #     v = v.asnumpy()

        v = v.T  # RCOPT

        x, sz_roll = unroll_dim(v, self.ndim + 1)
        x = m_reshape(
            x, new_shape=tuple([np.prod(self.sz)] + list(x.shape[self.ndim :]))
        )

        r_idx = self.basis_coords["r_idx"]
        ang_idx = self.basis_coords["ang_idx"]
        mask = m_flatten(self.basis_coords["mask"])

        ind = 0
        ind_radial = 0
        ind_ang = 0

        v = np.zeros(shape=tuple([self.count] + list(x.shape[1:])), dtype=v.dtype)
        for ell in range(0, self.ell_max + 1):
            k_max = self.k_max[ell]
            idx_radial = ind_radial + np.arange(0, k_max)
            # include the normalization factor of angular part
            ang_nrms = self.angular_norms[idx_radial]
            radial = self._precomp["radial"][:, idx_radial]
            radial = radial / ang_nrms

            sgns = (1,) if ell == 0 else (1, -1)
            for _ in sgns:
                ang = self._precomp["ang"][:, ind_ang]
                ang_radial = np.expand_dims(ang[ang_idx], axis=1) * radial[r_idx]
                idx = ind + np.arange(0, k_max)
                v[idx] = ang_radial.T @ x[mask]
                ind += len(idx)
                ind_ang += 1

            ind_radial += len(idx_radial)

        v = roll_dim(v, sz_roll)
        return v.T  # RCOPT

    def to_complex(self, coef):
        """
        Return complex valued representation of coefficients.
        This can be useful when comparing or implementing methods
        from literature.
        There is a corresponding method, to_real.
        :param coef: Coefficients from this basis.
        :return: Complex coefficent representation from this basis.
        """

        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        if coef.dtype not in (np.float64, np.float32):
            raise TypeError("coef provided to to_complex should be real.")

        # Pass through dtype precions, but check and warn if mismatched.
        dtype = complex_type(coef.dtype)
        # if coef.dtype != self.dtype:
            # logger.warning(
            #     f"coef dtype {coef.dtype} does not match precision of basis.dtype {self.dtype}, returning {dtype}."
            # )

        # Return the same precision as coef
        imaginary = dtype(1j)

        ccoef = np.zeros((coef.shape[0], self.complex_count), dtype=dtype)

        ind = 0
        idx = np.arange(self.k_max[0], dtype=int)
        ind += np.size(idx)

        ccoef[:, idx] = coef[:, idx]

        for ell in range(1, self.ell_max + 1):
            idx = ind + np.arange(self.k_max[ell], dtype=int)
            ccoef[:, idx] = (
                coef[:, self._pos[idx]] - imaginary * coef[:, self._neg[idx]]
            ) / 2.0

            ind += np.size(idx)

        return ccoef

    def to_real(self, complex_coef):
        """
        Return real valued representation of complex coefficients.
        This can be useful when comparing or implementing methods
        from literature.
        There is a corresponding method, to_complex.
        :param complex_coef: Complex coefficients from this basis.
        :return: Real coefficent representation from this basis.
        """
        if complex_coef.ndim == 1:
            complex_coef = complex_coef.reshape(1, -1)

        if complex_coef.dtype not in (np.complex128, np.complex64):
            raise TypeError("coef provided to to_real should be complex.")

        # Pass through dtype precions, but check and warn if mismatched.
        dtype = real_type(complex_coef.dtype)
        # if dtype != self.dtype:
        #     logger.warning(
        #         f"Complex coef dtype {complex_coef.dtype} does not match precision of basis.dtype {self.dtype}, returning {dtype}."
        #     )

        coef = np.zeros((complex_coef.shape[0], self.count), dtype=dtype)

        ind = 0
        idx = np.arange(self.k_max[0], dtype=int)
        ind += np.size(idx)
        ind_pos = ind

        coef[:, idx] = complex_coef[:, idx].real

        for ell in range(1, self.ell_max + 1):
            idx = ind + np.arange(self.k_max[ell], dtype=int)
            idx_pos = ind_pos + np.arange(self.k_max[ell], dtype=int)
            idx_neg = idx_pos + self.k_max[ell]

            c = complex_coef[:, idx]
            coef[:, idx_pos] = 2.0 * np.real(c)
            coef[:, idx_neg] = -2.0 * np.imag(c)

            ind += np.size(idx)
            ind_pos += 2 * self.k_max[ell]

        return coef

    """
    Rotation invariant distance.

    u1, u2: two images to compare.
    M: number of angle to test to compute the distance.
    """
    # def distance(self,u1, u2, M=360):
    #     c1 = self.evaluate_t(u1)
    #     c2 = self.evaluate_t(u2)
    #     # c1 = self.expand(u1)
    #     # c2 = self.expand(u2)

    #     if len(u1.shape)==2:
    #         c1 = c1[None,:]
    #         c2 = c2[None,:]
    #     # Nim = u1.shape[0]

    #     c1_comp = self.to_complex(c1)
    #     c2_comp = self.to_complex(c2)
    #     rad = 2*np.pi/M*np.arange(M)
    #     rot = np.exp(1j * self.complex_angular_indices[None,:] * rad[:,None])

    #     tmp = c1_comp*np.conj(c2_comp)
    #     dist = np.abs(np.matmul(rot,tmp.T))
    #     m_list = np.argmax(dist,axis=0)
        
    #     # c2_rot = self.to_real(c2_comp * np.exp(-1j * self.complex_angular_indices[None,None,:] * rad[None,:,None]))
    #     # dist = np.linalg.norm(c2_rot-c1[None,:],axis=1)
    #     # m = np.argmin(dist)
    #     return dist[m_list], rad[m_list]

    def distance_matrix(self,u1, u2, M=360):
        c1 = self.evaluate_t(u1)
        c2 = self.evaluate_t(u2)

        if len(u1.shape)==2:
            c1 = c1[None,:]
            c2 = c2[None,:]

        c1_comp = self.to_complex(c1)
        c2_comp = self.to_complex(c2)
        rad = 2*np.pi/M*np.arange(M)
        
        norm_ = (np.linalg.norm(c1_comp,axis=1)**2)[:,None]
        tmp = np.conj(c2_comp)
        dist = np.zeros((u1.shape[0],u2.shape[0]))
        angles = np.zeros((u1.shape[0],u2.shape[0]))
        for m in tqdm(range(M)):
            rot = np.exp(1j * self.complex_angular_indices[None,:] * rad[m])
            dist_ = np.abs(np.matmul(c1_comp*rot,tmp.T))/norm_
            dist = np.maximum(dist_,dist)
            angles[dist==dist_] = rad[m]
        return 1-dist, angles


    def Knn_mat(self, u1, K=10, M=90, verbose=False):
        ngpus = faiss.get_num_gpus()

        t = time.time()
        # Decompose using Fourier-Bessel expansion
        c1 = self.evaluate_t(u1)
        if len(u1.shape)==2:
            c1 = c1[None,:]
        c1_comp = self.to_complex(c1)
        rad = 2*np.pi/M*np.arange(M)
        if verbose:
            print("Decomposition: {0}".format(time.time()-t))

        t = time.time()
        exp_rot = np.exp(1j * self.complex_angular_indices[None,:] * rad[:,None])
        mult = exp_rot[None,:,:]*c1_comp[:,None,:]
        mult = mult.reshape(c1_comp.shape[0]*exp_rot.shape[0],-1)
        mult = np.concatenate((np.real(mult),np.imag(mult)),axis=1)
        if verbose:
            print("Dot-wise multiplication: {0}".format(time.time()-t))

        t = time.time()
        index = faiss.IndexFlatL2(mult.shape[1])
    
        if ngpus>1:
            res = faiss.StandardGpuResources()
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            gpu_index_flat = index
        gpu_index_flat.add(mult.reshape(mult.shape[0],-1).copy(order='C').astype("float32"))
        dist_, ind_ = gpu_index_flat.search(mult.reshape(mult.shape[0],-1).copy(order='C').astype("float32"), K+M)
        if verbose:
            print("Compute K-NN: {0}".format(time.time()-t))

        t = time.time()
        ## Remove neighbors that correspond to the same image
        ind_true = np.zeros((c1_comp.shape[0],K))
        dist_true = np.zeros((c1_comp.shape[0],K))
        for k in tqdm(range(c1_comp.shape[0])):
            ind_tmp = ind_[k*rad.shape[0]]
            dist_tmp = dist_[k*rad.shape[0]]
            idx = (ind_tmp >= k*rad.shape[0])*(ind_tmp < (k+1)*rad.shape[0])
            ind_tmp = ind_tmp[~idx]
            dist_tmp = dist_tmp[~idx]
            ind_true[k] = ind_tmp[:K]
            dist_true[k] = dist_tmp[:K]
        
        ind = np.array(ind_true//M,dtype=int)
        dist=dist_true
        angles = np.zeros_like(dist_true)
        for i in range(dist_true.shape[0]):
            for j in range(dist_true.shape[1]):
                angles[i,j] = rad[int(ind_true[i,j]%M)]
        if verbose:
            print("Post-process: {0}".format(time.time()-t))

        return dist, ind, angles

    """
    Similar method as the one above, but requiere less ram. It is however slower to run.
    """
    def Knn_mat_reduce_ram(self, u1, K=10, M=360, verbose=False):
        ngpus = faiss.get_num_gpus()

        t = time.time()
        # Decompose using Fourier-Bessel expansion
        c1 = self.evaluate_t(u1)
        if len(u1.shape)==2:
            c1 = c1[None,:]
        c1_comp = self.to_complex(c1)
        c1_conc = np.concatenate((np.real(c1_comp),np.imag(c1_comp)),axis=1)
        rad = 2*np.pi/M*np.arange(M)
        if verbose:
            print("Decomposition: {0}".format(time.time()-t))


        index = faiss.IndexFlatL2(c1_conc.shape[1])
        if ngpus>1:
            res = faiss.StandardGpuResources()
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            gpu_index_flat = index
        gpu_index_flat.add(c1_conc.reshape(c1_conc.shape[0],-1).copy(order='C').astype("float32"))

        dist = 1e16*np.ones((u1.shape[0],K))
        ind = np.arange(K)[None,:]
        ind = ind.repeat(u1.shape[0],0)
        angles = np.zeros((u1.shape[0],K))
        for m in range(M):
            exp_rot = np.exp(1j * self.complex_angular_indices * rad[m])
            # import ipdb; ipdb.set_trace()
            mult = exp_rot[None,:]*c1_comp
            mult = np.concatenate((np.real(mult),np.imag(mult)),axis=1)
            dist_, ind_ = index.search(mult.astype("float32"), K)
            # import ipdb; ipdb.set_trace()
            for k in range(K):
                ## Update indexes already here
                ind_k = ind_[:,k:k+1]
                ii_dist_ = np.logical_and((ind==ind_k),(dist_[:,k:k+1]<dist)) # Indexes already selected with lower distance value
                angles[ii_dist_] = rad[m] # Update angle for index already here but with now smaller dist value
                dist[ii_dist_] = dist_[ii_dist_] # update the distance for indexes already in the K-NN

                dist_k = dist_[:,k:k+1] # new candidate distances
                ii_new = (dist_k<dist) # indices of new neighbors
                ind_upd = np.logical_and((ii_new.sum(1)>0),(ind!=ind_k).sum(1)==K) # index to update
                ind_ax1 = np.argmax(dist[ind_upd]*(ii_new[ind_upd]*1.),1) # Among the existing distance larger than the new one, select the largest one
                dist[ind_upd,ind_ax1] = dist_k[ind_upd,0]
                angles[ind_upd,ind_ax1] = rad[m]
                ind[ind_upd,ind_ax1] = ind_k[ind_upd,0]
                    
                
        return dist, ind, angles


    # def calculate_bispectrum(
    #     self, coef, flatten=False, filter_nonzero_freqs=False, freq_cutoff=None
    # ):
    #     """
    #     Calculate bispectrum for a set of coefs in this basis.
    #     The Bispectum matrix is of shape:
    #         (count, count, unique_radial_indices)
    #     where count is the number of complex coefficients.
    #     :param coef: Coefficients representing a (single) image expanded in this basis.
    #     :param flatten: Optionally extract symmetric values (tril) and then flatten.
    #     :param freq_cutoff: Truncate (zero) high k frequecies above (int) value, defaults off (None).
    #     :return: Bispectum matrix (complex valued).
    #     """

    #     # Bispectrum implementation expects the complex representation of coefficients.
    #     complex_coef = self.to_complex(coef)

    #     return super().calculate_bispectrum(
    #         complex_coef,
    #         flatten=flatten,
    #         filter_nonzero_freqs=filter_nonzero_freqs,
    #         freq_cutoff=freq_cutoff,
    #     )

    def rotate(self, im, radians, refl=None):
        """
        Returns coefs rotated by `radians`.
        :param coef: Basis coefs.
        :param radians: Rotation in radians.
        :param refl: Optional reflect image (bool)
        :return: rotated coefs.
        """

    #     # Base class rotation expects complex representation of coefficients.
    #     #  Convert, rotate and convert back to real representation.
    #     return self.to_real(super().rotate(self.to_complex(coef), radians, refl))

    # def rotate(self, complex_coef, radians, refl=None):
    #     """
    #     Returns complex coefs rotated by `radians`.
    #     :param complex_coef: Basis coefs (in complex representation).
    #     :param radians: Rotation in radians.
    #     :param refl: Optional reflect image (about y=x) (bool)
    #     :return: rotated (complex) coefs.
    #     """

        coef = self.evaluate_t(im)
        complex_coef = self.to_complex(coef)

        # Covert radians to a broadcastable shape
        if isinstance(radians, np.ndarray):
            if len(radians) != len(complex_coef):
                raise RuntimeError(
                    "`rotate` call `radians` length cannot broadcast with"
                    f" `complex_coef` {len(complex_coef)} != {len(radians)}"
                )
            radians = radians.reshape(-1, 1)
        # else: radians can be a constant

        ks = self.complex_angular_indices
        assert len(ks) == complex_coef.shape[-1]

        # refl
        if refl is not None:
            if isinstance(refl, np.ndarray):
                assert len(refl) == len(complex_coef)
            # else: refl can be a constant
            # get the coefs corresponding to -ks , aka "ells"
            complex_coef[refl] = np.conj(complex_coef[refl])

        complex_coef[:] = complex_coef * np.exp(-1j * ks * radians)
        im_rot = self.evaluate(self.to_real(complex_coef))
        return np.squeeze(im_rot)
