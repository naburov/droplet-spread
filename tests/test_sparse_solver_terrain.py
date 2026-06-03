"""Test terrain Laplacian in sparse solver: flat f(x)=0 must match Cartesian."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solvers.sparse_solver import SparseSolverWrapper


def test_flat_terrain_matches_cartesian():
    """With f_1=0, f_2=0 terrain stencil reduces to Cartesian 5-point Laplacian."""
    Nx, Ny = 8, 6
    dx, dy = 0.1, 0.12
    # Cartesian solver (no terrain)
    cart = SparseSolverWrapper(Nx, Ny, dx, dy, backend="scipy")
    cart.set_top_boundary_condition("neumann")
    cart.set_bottom_boundary_condition("neumann")
    cart.set_left_boundary_condition("neumann")
    cart.set_right_boundary_condition("neumann")
    cart.create_sparse_matrix()
    A_cart = cart.A.toarray()
    # Terrain solver with f=0 (flat)
    f1_flat = np.zeros((Nx, Ny))
    f2_flat = np.zeros((Nx, Ny))
    terrain = SparseSolverWrapper(Nx, Ny, dx, dy, backend="scipy")
    terrain.set_top_boundary_condition("neumann")
    terrain.set_bottom_boundary_condition("neumann")
    terrain.set_left_boundary_condition("neumann")
    terrain.set_right_boundary_condition("neumann")
    terrain.set_terrain(f1_flat, f2_flat)
    A_terrain_flat = terrain.A.toarray()
    # With f=0, _use_terrain() is False so we build Cartesian; set_terrain(zeros) still triggers Cartesian
    # So terrain solver with zeros should build Cartesian (same as cart)
    np.testing.assert_allclose(A_terrain_flat, A_cart, rtol=1e-10, atol=1e-12)
    # Same solution for a test RHS
    rhs = np.random.randn(Nx, Ny).astype(np.float64)
    rhs = rhs - np.mean(rhs)
    sol_cart = cart.solve(rhs)
    sol_terrain = terrain.solve(rhs)
    np.testing.assert_allclose(sol_terrain, sol_cart, rtol=1e-9, atol=1e-9)
    print("test_flat_terrain_matches_cartesian: OK (flat f=0 matches Cartesian)")


def test_terrain_nonflat_different_from_cartesian():
    """With non-zero f_1 the terrain matrix differs from Cartesian."""
    Nx, Ny = 6, 4
    dx, dy = 0.1, 0.12
    cart = SparseSolverWrapper(Nx, Ny, dx, dy, backend="scipy")
    cart.set_top_boundary_condition("neumann")
    cart.set_bottom_boundary_condition("neumann")
    cart.set_left_boundary_condition("neumann")
    cart.set_right_boundary_condition("neumann")
    cart.create_sparse_matrix()
    A_cart = cart.A.toarray()
    f1 = np.full((Nx, Ny), 0.2)  # tilted
    f2 = np.zeros((Nx, Ny))
    terrain = SparseSolverWrapper(Nx, Ny, dx, dy, backend="scipy")
    terrain.set_top_boundary_condition("neumann")
    terrain.set_bottom_boundary_condition("neumann")
    terrain.set_left_boundary_condition("neumann")
    terrain.set_right_boundary_condition("neumann")
    terrain.set_terrain(f1, f2)
    A_terrain = terrain.A.toarray()
    diff = np.abs(A_terrain - A_cart)
    assert np.any(diff > 1e-10), "Terrain matrix with f'≠0 should differ from Cartesian"
    print("test_terrain_nonflat_different_from_cartesian: OK")


if __name__ == "__main__":
    test_flat_terrain_matches_cartesian()
    test_terrain_nonflat_different_from_cartesian()
    print("All terrain sparse solver tests passed.")
