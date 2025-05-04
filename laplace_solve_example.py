import numpy as np
from scipy.sparse import diags, kron, identity

def build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy, bc_type='dirichlet'):
    """
    Constructs the 2D Laplacian matrix using Kronecker product with variable spatial steps
    in x and y directions, supporting different boundary conditions.
    
    Parameters:
        Nx (int): Number of interior grid points in x-direction
        Ny (int): Number of interior grid points in y-direction
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        bc_type (str): 'dirichlet' or 'neumann'
    
    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix of shape (Nx*Ny, Nx*Ny)
    """
    # 1D Laplacian for x-direction
    main_diag_x = -2 * np.ones(Nx) / (dx**2)
    off_diag_x = np.ones(Nx - 1) / (dx**2)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1], shape=(Nx, Nx))
    
    # 1D Laplacian for y-direction
    main_diag_y = -2 * np.ones(Ny) / (dy**2)
    off_diag_y = np.ones(Ny - 1) / (dy**2)
    Ty = diags([off_diag_y, main_diag_y, off_diag_y], [-1, 0, 1], shape=(Ny, Ny))
    
    # Apply boundary conditions
    Tx = Tx.tolil()
    Tx[0, 1] = 2 / (dx**2)  # Left boundary mirror
    Tx[-1, -2] = 2 / (dx**2)  # Right boundary mirror
    Tx = Tx.tocsr()
        
    Ty = Ty.tolil()
    Ty[0, 0] = 1.0
    Ty[0, 1] = 0.0  
    
    Ty[-1, -1] = 1.0
    Ty[-1, -2] = 0.0
    Ty = Ty.tocsr()
    
    # Create identity matrices
    Ix = identity(Nx)
    Iy = identity(Ny)
    
    # Combine using Kronecker products to create 2D Laplacian
    A = kron(Iy, Tx) + kron(Ty, Ix)
    
    return A

if __name__ == "__main__":
    Nx = 4
    Ny = 4
    dx = 1.0
    dy = 1.0
    bc_type = 'neumann'
    A = build_2d_laplacian_matrix_with_variable_steps(Nx, Ny, dx, dy, bc_type)
    print(A.toarray())
