# Corner behavior (bottom-left and bottom-right)

For `contact_angle_60.json` the domain corners where **bottom** (wall) meets **left/right** (atmosphere) are treated as follows.

## Velocity (staggered)

- **Bottom**: navier_slip → `v_face[:,0]=0`, `u_face[:,0] = u_face[:,1]*λ/(λ+dy)`.
- **Left/Right**: Neumann → `u_face[0,:]=u_face[1,:]`, `v_face[0,:]=v_face[1,:]` (and similarly for right).

BCs are applied **per edge**; there is no separate “corner” step. So:

- The **bottom face** of the corner cell is set by the bottom BC (v=0, u from slip).
- The **left/right face** of the corner cell is set by the left/right BC (extrapolate from interior).

Which value “wins” at the corner depends on **order of application** (bottom, then left, then right in the code). So the corner cell’s left face is overwritten by the left BC. That is consistent with “wall along bottom, stress-free at sides”: the corner is effectively the end of the wall, with zero normal velocity on the wall and extrapolated tangential values from the side.

**Collocated** path: corners are only forced to zero when at least one side is **no_slip**. Here bottom is navier_slip and left/right are Neumann, so corners are **not** zeroed; they keep the value from the last-applied edge (left/right extrapolation).

## Pressure / PPE (Poisson solvers)

The 2D Laplacian is built as **A = Iy⊗Tx + Ty⊗Ix** (Kronecker products of 1D matrices). So each grid point, including corners, has **one** row that combines:

- The 1D BC for **left/right** (from Tx): Dirichlet → row “P = rhs”.
- The 1D BC for **bottom/top** (from Ty): Neumann → row “∂P/∂y” stencil.

So at **(0,0)** (bottom-left):

- **Left** contributes a Dirichlet-like term (from Tx).
- **Bottom** contributes a Neumann-like term (from Ty).

The **single** equation for (0,0) is therefore a **mix** of “P = 0” (left) and “∂P/∂n = 0” (bottom). So **P(0,0) is not strictly pinned to 0**; it is coupled to the neighbor P(0,1). In practice P(0,0) stays close to 0 but can deviate slightly, which can give a **small local pressure gradient** near the corner.

**RHS**: We set `rhs[0,:]=0` (left) and do not overwrite `rhs[:,0]` for bottom (Neumann). So `rhs[0,0]=0`. The matrix row for (0,0) is still the mixed left+bottom row, so the solution at the corner is determined by that combined equation, not by a pure “P(0,0)=0” row.

## Phase field

- **Bottom**: contact_angle (and optionally Cox–Voinov) on the full row j=0, including i=0 and i=Nx-1.
- **Left/Right**: Neumann (zero normal derivative) on the full column.

So at the corner cells (0,0) and (Nx-1,0), the **phase** gets:

- Contact angle condition from the bottom (normal derivative of φ from the wall).
- Neumann from the side (φ copied or extrapolated in x).

No extra “corner” rule is applied; the corner is part of both the bottom and the side. That is usually acceptable because the contact line rarely sits exactly on the discrete corner.

## Summary

| Quantity   | At bottom-left / bottom-right |
|-----------|--------------------------------|
| **Velocity** | One edge (bottom) sets wall-normal and slip; the other (left/right) sets extrapolation. Order of application decides the stored face/cell value. No explicit corner zeroing (no_slip not used). |
| **Pressure** | Single Poisson row mixes Dirichlet (left/right) and Neumann (bottom). Corner value not strictly P=0; can have a small deviation and thus a **small spurious gradient** near the corner. |
| **Phase**   | Contact angle on bottom row; Neumann on left/right columns. No special corner treatment. |

If you see **corner artifacts** (e.g. small vortices or pressure spikes), options are:

1. **Pressure**: Add explicit corner handling in the Poisson solver so that bottom-left and bottom-right get a pure Dirichlet row (e.g. P=0) when both sides are open.
2. **Velocity**: Add an explicit corner rule (e.g. average of the two boundary prescriptions or prefer wall normal=0 at the corner).
3. **Smoothing**: Slight smoothing or filtering of pressure/velocity in a 1–2 cell band near corners (heuristic).
