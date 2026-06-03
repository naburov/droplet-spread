# Streamlines: grid, coordinates, and indexing

## 1. PyVista path (`pyvista_utils.py`)

### 1.1 Vertex grid (mesh for streamlines)

Built in `mesh_geometry_for_plot(Nx, Ny, Lx, Ly, dx, dy, geometry)`:

- **1D arrays**
  - `x_verts = np.linspace(0, Lx, Nx + 1)`  → shape `(Nx+1,)`, x at vertices
  - `eta_verts = np.linspace(0, Ly, Ny + 1)` → shape `(Ny+1,)`, eta at vertices

- **meshgrid with `indexing="xy"`**
  - `X_verts, Eta_verts = np.meshgrid(x_verts, eta_verts, indexing="xy")`
  - With `indexing="xy"`: first dimension = second input (eta), second = first input (x).
  - So:
    - `X_verts.shape = (Ny+1, Nx+1)`, `Eta_verts.shape = (Ny+1, Nx+1)`
    - `X_verts[row, col] = x_verts[col]`  → x varies along **columns**
    - `Eta_verts[row, col] = eta_verts[row]` → eta varies along **rows**
  - Convention: **row index = eta index**, **column index = x index**.

- **Physical y**
  - `Y_verts = Eta_verts + f_verts[np.newaxis, :]` with `f_verts = f(x_verts)`
  - So `Y_verts[row, col] = eta_verts[row] + f(x_verts[col])` = physical y at (x, eta).
  - Mapping: **(x, eta) → (x, y)** with **y = eta + f(x)**.

So the PyVista structured grid uses:

- **Grid arrays**: `X_verts`, `Y_verts`, both shape **(Ny+1, Nx+1)**.
- **Indexing**: `[row, col]` = `[eta_idx, x_idx]` with `eta_idx` = 0..Ny, `x_idx` = 0..Nx.
- **Coordinates**: physical **(x, y)** with x horizontal, y = eta + f(x).

### 1.2 Velocity on the vertex grid

In `_add_streamlines`:

- **Input**: `U_cc` shape **(Nx, Ny, 2)** (cell-centered): `U_cc[i, j, :]` = (u_x, u_eta) at cell (x_i, eta_j).

- **Cell-centered → vertex**
  - `cell_centered_to_point_2d(Z_cc)` takes `(Nx, Ny)` and returns **(Nx+1, Ny+1)**.
  - Convention: **first index = x**, **second index = eta**.
  - So `Ux[i, j]` = u_x at vertex **(x_idx=i, eta_idx=j)**, i ∈ 0..Nx, j ∈ 0..Ny.

- **Physical y-component for streamlines**
  - In (x, eta) we have (u_x, u_eta). In physical (x, y): **u_y = u_eta + f'(x) u_x**.
  - `x_verts = X_verts[0, :]` → x at each column, shape (Nx+1,).
  - `Uy = Uy_eta + f_prime[:, np.newaxis] * Ux` so `Uy[i, j]` = u_y at (x_i, eta_j).

- **Matching grid point order**
  - PyVista grid points are ordered like `X_verts.ravel(order="C")`: row 0 (eta=0), then row 1, …
  - So point order is **(eta_idx, x_idx)**.
  - We assign: `grid["vec"] = np.column_stack([Ux.T.ravel(order="C"), Uy.T.ravel(order="C"), 0])`.
  - `Ux.T` has shape (Ny+1, Nx+1) with `Ux.T[eta_idx, x_idx] = Ux[x_idx, eta_idx]`, so after ravel the k-th point (eta_idx, x_idx) gets **(Ux[x_idx, eta_idx], Uy[x_idx, eta_idx])** = velocity at (x_verts[x_idx], Y_verts[eta_idx, x_idx]). So grid coordinates and velocity indexing are consistent.

### 1.3 Boundary masking (PyVista)

- Bottom row (`j=0` / `eta_idx=0`): `Ux[:, 0] = 0`, `Uy_eta[:, 0] = 0` so streamlines stop at the surface.
- Top row: `Ux[:, -1] = 0`, `Uy_eta[:, -1] = 0` so streamlines do not cross/emanate from the top.

---

## 2. Matplotlib path

### 2.1 `plot_layout.prepare_joint_plot_data`

- **Cell-centered 1D**
  - `x = np.linspace(0, Lx, Nx)`   → Nx points (cell centers)
  - `eta = np.linspace(0, Ly, Ny)`  → Ny points (cell centers)

- **meshgrid**
  - `X, Eta = np.meshgrid(x, eta)` (default is `indexing="xy"` in recent NumPy).
  - So `X.shape = Eta.shape = (Ny, Nx)`, with **row = eta**, **col = x**.

- **Velocity**
  - `U_masked` shape **(Nx, Ny, 2)** with `U_masked[i, j, :]` at (x[i], eta[j]).

- **streamplot**
  - `ax.streamplot(data["x"], data["eta"], data["U_masked"][..., 0].T, data["U_masked"][..., 1].T, ...)`
  - Grid: (x[i], eta[j]). Components passed as **(Ny, Nx)** via `.T` so `u[j, i] = U_masked[i, j, 0]` at (x[i], eta[j]). So streamplot is in **(x, eta)** (computational), not physical (x, y).

### 2.2 `staggered_flow_logging.save_frame`

- **Cell centers**: `x = (np.arange(Nx) + 0.5)*dx`, `y = (np.arange(Ny) + 0.5)*dy` (here `y` is eta).
- **Velocity**: `uc`, `vc` shape (Nx, Ny) at (x[i], y[j]).
- **streamplot**: `ax3.streamplot(x, y, uc.T, vc.T, ...)` so (x, y) grid with u[j,i]=uc[i,j]. Same (x, eta) convention; `imshow(..., extent=[0, Lx, 0, Ly])` so vertical axis is 0..Ly (eta).

---

## 3. Summary table

| Location              | Grid type     | meshgrid indexing | Grid shape  | Index [1st, 2nd] | Coordinates   |
|-----------------------|---------------|-------------------|-------------|------------------|---------------|
| PyVista vertex        | vertices      | `"xy"`            | (Ny+1, Nx+1)| [eta, x]         | physical (x, y), y=η+f(x) |
| PyVista velocity (Ux/Uy) | vertices    | —                 | (Nx+1, Ny+1)| [x, eta]         | same points, assigned via .T |
| plot_layout (X, Eta)  | cell centers  | default "xy"      | (Ny, Nx)    | [eta, x]         | (x, eta)       |
| plot_layout streamplot| cell centers | —                 | u,v: (Ny,Nx)| —                | (x, eta)       |
| staggered_flow_logging| cell centers | —                 | (Nx, Ny)    | [x, eta]         | (x, eta), extent (0,Lx,0,Ly) |

---

## 4. NumPy `meshgrid` reminder

- `indexing="xy"`: first dimension = second input (y/eta), second = first input (x). So `X[i,j]=x[j]`, `Y[i,j]=y[i]`. Shape (len(y), len(x)).
- `indexing="ij"`: first dimension = first input (x), second = second (y). So `X[i,j]=x[i]`, `Y[i,j]=y[j]`. Shape (len(x), len(y)).

PyVista vertex grid and plot_layout use **"xy"** so that **[row, col] = [eta_idx, x_idx]**.
