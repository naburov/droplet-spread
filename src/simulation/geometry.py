"""
Geometry representation for simulation.

Surface is described by y = f(x). Base class Geometry and subclasses provide
f(x), f'(x), f''(x) and JAX-ready tensors f_1_grid, f_2_grid, h_bottom.
Flat surface: f = f' = f'' = 0. When passing to JAX, use geometry.f_1_grid, geometry.f_2_grid.
"""

import numpy as np
import jax.numpy as jnp
from typing import Union


def _expand_to_grid(val: Union[float, jnp.ndarray], Nx: int, Ny: int) -> jnp.ndarray:
    """Expand f' or f'' to (Nx, Ny). Scalar -> full; (Nx,) -> broadcast along y."""
    arr = jnp.asarray(val)
    if arr.ndim == 0 or arr.size == 1:
        return jnp.full((Nx, Ny), float(jnp.ravel(arr)[0]), dtype=jnp.float64)
    return jnp.broadcast_to(arr[:, None], (Nx, Ny))


def _interp_or_grid(
    x: Union[float, jnp.ndarray],
    xp: jnp.ndarray,
    fp: jnp.ndarray,
) -> Union[float, jnp.ndarray]:
    """Evaluate fp at x. x: scalar or array; xp, fp: 1D."""
    xp_1d, fp_1d = jnp.ravel(jnp.asarray(xp)), jnp.ravel(jnp.asarray(fp))
    scalar = isinstance(x, (int, float)) or (getattr(x, "ndim", None) == 0) or (getattr(x, "size", 0) == 1)
    if scalar:
        return float(jnp.interp(float(x), jnp.array(xp_1d), jnp.array(fp_1d)))
    x_arr = jnp.asarray(x)
    return jnp.reshape(jnp.interp(jnp.ravel(x_arr), xp_1d, fp_1d), x_arr.shape)


def _finite_diff_1d(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Central diff interior, one-sided at boundaries. 1D."""
    df = jnp.zeros_like(f)
    df = df.at[1:-1].set((f[2:] - f[:-2]) / (2.0 * dx))
    df = df.at[0].set((f[1] - f[0]) / dx).at[-1].set((f[-1] - f[-2]) / dx)
    return df


def _finite_diff_2d(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Second derivative: central interior, one-sided at boundaries. 1D."""
    d2f = jnp.zeros_like(f)
    d2f = d2f.at[1:-1].set((f[2:] - 2.0 * f[1:-1] + f[:-2]) / (dx * dx))
    d2f = d2f.at[0].set((f[2] - 2.0 * f[1] + f[0]) / (dx * dx)).at[-1].set((f[-1] - 2.0 * f[-2] + f[-3]) / (dx * dx))
    return d2f


def _as_scalar_if_constant(arr: jnp.ndarray, rtol: float = 1e-9, atol: float = 1e-12) -> Union[float, jnp.ndarray]:
    """Constant array -> scalar; else return arr."""
    a = np.asarray(arr).ravel()
    if not a.size or np.allclose(a, a[0], rtol=rtol, atol=atol):
        return float(a[0]) if a.size else 0.0
    return arr


class Geometry:
    """Base class for bottom surface geometry y = f(x).

    Subclasses provide f_1_grid, f_2_grid, h_bottom (JAX tensors) and has_geometry.
    """

    def __init__(self, Nx: int, Ny: int):
        self.Nx = Nx
        self.Ny = Ny

    @property
    def f_1_grid(self) -> jnp.ndarray:
        """f'(x) on (Nx, Ny). For JAX diff operators."""
        raise NotImplementedError

    @property
    def f_2_grid(self) -> jnp.ndarray:
        """f''(x) on (Nx, Ny). For JAX diff operators."""
        raise NotImplementedError

    @property
    def h_bottom(self) -> jnp.ndarray:
        """Bottom surface height h(x) at grid, shape (Nx,). For flat, zeros."""
        raise NotImplementedError

    @property
    def has_geometry(self) -> bool:
        """True if non-flat surface (any f(x) != 0)."""
        raise NotImplementedError

    def f(self, x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Surface height f(x). Override in subclasses if needed."""
        return _interp_or_grid(x, self._x_grid, self._f)

    def f_1(self, x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """First derivative f'(x)."""
        return self._deriv_at_x(x, self._f_1)

    def f_2(self, x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Second derivative f''(x)."""
        return self._deriv_at_x(x, self._f_2)

    def _deriv_at_x(self, x: Union[float, jnp.ndarray], f_val: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        if isinstance(f_val, (int, float)):
            return f_val
        arr = jnp.asarray(f_val)
        if arr.ndim == 0 or arr.size == 1:
            return float(jnp.ravel(arr)[0])
        return _interp_or_grid(x, self._x_grid, f_val)


class FlatGeometry(Geometry):
    """Flat surface: f = f' = f'' = 0."""

    def __init__(self, Nx: int, Ny: int):
        super().__init__(Nx, Ny)
        self._x_grid = jnp.arange(Nx, dtype=jnp.float64)
        self._f = jnp.zeros(Nx, dtype=jnp.float64)
        self._f_1 = 0.0
        self._f_2 = 0.0
        self._f_1_grid = jnp.zeros((Nx, Ny), dtype=jnp.float64)
        self._f_2_grid = jnp.zeros((Nx, Ny), dtype=jnp.float64)

    @property
    def f_1_grid(self) -> jnp.ndarray:
        return self._f_1_grid

    @property
    def f_2_grid(self) -> jnp.ndarray:
        return self._f_2_grid

    @property
    def h_bottom(self) -> jnp.ndarray:
        return self._f

    @property
    def has_geometry(self) -> bool:
        return False


class TiltedGeometry(Geometry):
    """Tilted plane y = slope * (x - x_origin). f' = slope, f'' = 0."""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float,
        Lx: float,
        Ly: float,
        angle_degrees: float = 10.0,
        origin: str = "bottom_left",
    ):
        super().__init__(Nx, Ny)
        theta = np.deg2rad(angle_degrees)
        slope = float(np.tan(theta))
        if origin == "bottom_right":
            x_origin = Lx
        elif origin == "center":
            x_origin = Lx / 2.0
        else:
            x_origin = 0.0
        x_grid = np.arange(Nx, dtype=np.float64) * dx
        f_at_grid = slope * (x_grid - x_origin)
        f_at_grid = np.clip(f_at_grid, 0.0, Ly)
        # Build from height arrays
        self._x_grid = jnp.asarray(x_grid)
        self._f = jnp.asarray(f_at_grid, dtype=jnp.float64)
        self._f_1 = slope
        self._f_2 = 0.0
        self._f_1_grid = _expand_to_grid(slope, Nx, Ny)
        self._f_2_grid = jnp.zeros((Nx, Ny), dtype=jnp.float64)

    @property
    def f_1_grid(self) -> jnp.ndarray:
        return self._f_1_grid

    @property
    def f_2_grid(self) -> jnp.ndarray:
        return self._f_2_grid

    @property
    def h_bottom(self) -> jnp.ndarray:
        return self._f

    @property
    def has_geometry(self) -> bool:
        return bool(np.any(np.abs(np.asarray(self._f)) > 1e-14))


class FromHeightGeometry(Geometry):
    """Arbitrary surface from f(x) at grid points. Computes f', f'' via finite differences."""

    def __init__(self, f_at_grid: jnp.ndarray, Ny: int, dx: float):
        Nx = int(f_at_grid.shape[0])
        super().__init__(Nx, Ny)
        self._x_grid = jnp.arange(Nx, dtype=jnp.float64) * dx
        self._f = jnp.asarray(f_at_grid, dtype=jnp.float64)
        f_1 = _finite_diff_1d(self._f, dx)
        f_2 = _finite_diff_2d(self._f, dx)
        f_1_stored = _as_scalar_if_constant(f_1)
        f_2_stored = _as_scalar_if_constant(f_2)
        self._f_1 = f_1_stored
        self._f_2 = f_2_stored
        self._f_1_grid = _expand_to_grid(f_1_stored, Nx, Ny)
        self._f_2_grid = _expand_to_grid(f_2_stored, Nx, Ny)

    @property
    def f_1_grid(self) -> jnp.ndarray:
        return self._f_1_grid

    @property
    def f_2_grid(self) -> jnp.ndarray:
        return self._f_2_grid

    @property
    def h_bottom(self) -> jnp.ndarray:
        return self._f

    @property
    def has_geometry(self) -> bool:
        return bool(jnp.any(jnp.abs(self._f) > 1e-14))


class HumpGeometry(Geometry):
    """Gaussian hump y = A * exp(-(x - x0)^2 / (2*sigma^2)) with analytical f', f''."""

    def __init__(self, Nx: int, Ny: int, dx: float, amplitude: float, sigma: float, center_x: float):
        super().__init__(Nx, Ny)
        self._x_grid = jnp.arange(Nx, dtype=jnp.float64) * dx
        x = self._x_grid
        s2 = sigma * sigma
        self._f = amplitude * jnp.exp(-((x - center_x) ** 2) / (2.0 * s2))
        # f' = -f * (x - center_x) / sigma^2
        self._f_1 = -self._f * (x - center_x) / s2
        # f'' = f * ((x - center_x)^2 / sigma^4 - 1/sigma^2)
        self._f_2 = self._f * ((x - center_x) ** 2 / (s2 * s2) - 1.0 / s2)
        self._f_1_grid = _expand_to_grid(self._f_1, Nx, Ny)
        self._f_2_grid = _expand_to_grid(self._f_2, Nx, Ny)

    @property
    def f_1_grid(self) -> jnp.ndarray:
        return self._f_1_grid

    @property
    def f_2_grid(self) -> jnp.ndarray:
        return self._f_2_grid

    @property
    def h_bottom(self) -> jnp.ndarray:
        return self._f

    @property
    def has_geometry(self) -> bool:
        return bool(jnp.any(jnp.abs(self._f) > 1e-14))


class SinusoidalGrooveGeometry(Geometry):
    """Periodic grooved substrate h(x) = offset + A * (1 - cos(2π n x / Lx)) / 2."""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        dx: float,
        Lx: float,
        amplitude: float,
        waves: float = 2.0,
        offset: float = 0.0,
        phase: float = 0.0,
    ):
        super().__init__(Nx, Ny)
        self._x_grid = jnp.arange(Nx, dtype=jnp.float64) * dx
        k = 2.0 * np.pi * float(waves) / float(Lx)
        arg = k * self._x_grid + float(phase)
        amp = float(amplitude)
        self._f = float(offset) + 0.5 * amp * (1.0 - jnp.cos(arg))
        self._f_1 = 0.5 * amp * k * jnp.sin(arg)
        self._f_2 = 0.5 * amp * k * k * jnp.cos(arg)
        self._f_1_grid = _expand_to_grid(self._f_1, Nx, Ny)
        self._f_2_grid = _expand_to_grid(self._f_2, Nx, Ny)

    @property
    def f_1_grid(self) -> jnp.ndarray:
        return self._f_1_grid

    @property
    def f_2_grid(self) -> jnp.ndarray:
        return self._f_2_grid

    @property
    def h_bottom(self) -> jnp.ndarray:
        return self._f

    @property
    def has_geometry(self) -> bool:
        return bool(jnp.any(jnp.abs(self._f) > 1e-14))


# Factory methods on base class (state.py uses Geometry.flat, Geometry.tilted, Geometry.hump)
Geometry.flat = staticmethod(lambda Nx, Ny, dy=None: FlatGeometry(Nx, Ny))
Geometry.tilted = staticmethod(
    lambda Nx, Ny, dx, dy, Lx, Ly, angle_degrees=10.0, origin="bottom_left": TiltedGeometry(
        Nx, Ny, dx, dy, Lx, Ly, angle_degrees=angle_degrees, origin=origin
    )
)
Geometry.from_height = staticmethod(
    lambda f_at_grid, Ny, dy, dx: FromHeightGeometry(f_at_grid, Ny, dx)
)
Geometry.hump = staticmethod(
    lambda Nx, Ny, dx, amplitude, sigma, center_x, dy=None: HumpGeometry(
        Nx, Ny, dx, amplitude, sigma, center_x
    )
)
Geometry.sinusoidal_groove = staticmethod(
    lambda Nx, Ny, dx, Lx, amplitude, waves=2.0, offset=0.0, phase=0.0: SinusoidalGrooveGeometry(
        Nx, Ny, dx, Lx, amplitude, waves=waves, offset=offset, phase=phase
    )
)
