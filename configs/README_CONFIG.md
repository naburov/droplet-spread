# 📋 **Configuration Files Documentation**

## 🎯 **Enhanced Configuration Structure**

The configuration files now include three major new sections for complete control over simulations:

### **1. 🚧 Boundary Conditions (`boundary_conditions`)**

Controls how the simulation behaves at domain boundaries:

```json
"boundary_conditions": {
    "pressure": {
        "top": "dirichlet",      // Fixed pressure value
        "bottom": "dirichlet",   // Fixed pressure value  
        "left": "dirichlet",     // Fixed pressure value
        "right": "dirichlet"     // Fixed pressure value
    },
    "velocity": {
        "top": "neumann",        // Zero gradient (open atmosphere)
        "bottom": "no_slip",     // Zero velocity (solid wall)
        "left": "periodic",      // Periodic boundary
        "right": "periodic"      // Periodic boundary
    },
    "phase_field": {
        "top": "neumann",        // Zero gradient
        "bottom": "contact_angle", // Prescribed contact angle
        "left": "periodic",      // Periodic boundary
        "right": "periodic"      // Periodic boundary
    }
}
```

**Available Boundary Types:**
- **`dirichlet`**: Fixed value at boundary
- **`neumann`**: Zero gradient (∂φ/∂n = 0)
- **`no_slip`**: Zero velocity (for velocity field)
- **`periodic`**: Periodic boundary conditions
- **`contact_angle`**: Prescribed contact angle (for phase field)

### **2. ⚙️ Solver Parameters (`solver_params`)**

Controls numerical solution methods and convergence:

```json
"solver_params": {
    "pressure_solver": {
        "backend": "pyamg",      // "pyamg" or "scipy"
        "accel": "bicgstab",     // "bicgstab", "cg", "gmres"
        "tol": 0.05,             // Convergence tolerance
        "maxiter": 10000         // Maximum iterations
    },
    "correction_solver": {
        "backend": "pyamg", 
        "accel": "bicgstab",
        "tol": 0.05,
        "maxiter": 10000
    },
    "divergence_threshold": 0.05,        // Max allowed divergence
    "max_correction_iterations": 100     // Max pressure correction steps
}
```

**Solver Backends:**
- **`pyamg`**: Algebraic multigrid (recommended for large problems)
- **`scipy`**: Direct sparse solver (faster for small problems)

**Acceleration Methods:**
- **`bicgstab`**: BiCGSTAB (recommended)
- **`cg`**: Conjugate Gradient
- **`gmres`**: Generalized Minimal Residual

### **3. 📊 Plotting Parameters (`plotting_params`)**

Controls visualization output:

```json
"plotting_params": {
    "figure_size": [18, 14],           // [width, height] in inches
    "dpi": 100,                        // Resolution (dots per inch)
    "colormap": "viridis",             // Color scheme
    "save_format": "png",              // Output format
    "show_velocity_vectors": true,     // Show velocity field
    "vector_density": 8,               // Vector spacing
    "vector_scale": 50,                // Vector length scaling
    "show_contours": true,             // Show contour lines
    "contour_levels": 20,              // Number of contour levels
    "show_colorbar": true,             // Show colorbar
    "title_fontsize": 16,              // Title font size
    "label_fontsize": 14,              // Axis label font size
    "tick_fontsize": 12                // Tick label font size
}
```

**Available Colormaps:**
- **`viridis`**: Perceptually uniform (default)
- **`plasma`**: High contrast
- **`coolwarm`**: Diverging colors
- **`jet`**: Classic rainbow
- **`RdYlBu`**: Red-Yellow-Blue diverging

## 🎨 **Configuration Presets**

### **Water Droplet (`config_water_droplet.json`)**
- **High resolution**: 150 DPI, 20×16 figure
- **Plasma colormap**: High contrast for water dynamics
- **Tight solver tolerance**: 0.01 for accuracy
- **Dense vectors**: 6 density, 30 scale

### **Air Bubble (`config_air_bubble.json`)**
- **Medium resolution**: 120 DPI, 16×12 figure  
- **Coolwarm colormap**: Diverging colors for bubble
- **Moderate solver tolerance**: 0.02 for stability
- **Sparse vectors**: 10 density, 40 scale

### **Template (`config_template.json`)**
- **Standard resolution**: 100 DPI, 18×14 figure
- **Viridis colormap**: Balanced visualization
- **Standard solver tolerance**: 0.05 for general use
- **Medium vectors**: 8 density, 50 scale

## 🔧 **Usage Examples**

### **High-Resolution Water Droplet**
```json
"plotting_params": {
    "figure_size": [24, 18],
    "dpi": 300,
    "colormap": "plasma"
}
```

### **Fast Solver for Testing**
```json
"solver_params": {
    "pressure_solver": {
        "backend": "scipy",
        "tol": 0.1,
        "maxiter": 1000
    }
}
```

### **No-Slip Bottom Wall**
```json
"boundary_conditions": {
    "velocity": {
        "bottom": "no_slip"
    }
}
```

### **Open Top Boundary**
```json
"boundary_conditions": {
    "velocity": {
        "top": "neumann"
    }
}
```

## 🎯 **Benefits**

1. **🎛️ Complete Control**: Fine-tune every aspect of simulation
2. **🔧 Easy Tuning**: Adjust parameters without code changes
3. **📊 Flexible Visualization**: Customize plots for different needs
4. **⚡ Performance Optimization**: Balance speed vs accuracy
5. **🎨 Visual Quality**: High-resolution outputs for publications
6. **🔬 Research Ready**: Professional configuration management

## 🚀 **Next Steps**

1. **Test different configurations** to find optimal settings
2. **Create custom presets** for specific research needs
3. **Experiment with boundary conditions** for different physics
4. **Tune solver parameters** for your hardware
5. **Customize plotting** for publication-quality figures

The enhanced configuration system makes your droplet spreading simulations **highly configurable** and **research-ready**! 🎉
