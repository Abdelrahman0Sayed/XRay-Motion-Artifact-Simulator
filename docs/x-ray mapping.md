# X-Ray Motion Artifact Simulator: Parallel Beam Implementation

This document details the mathematical models, execution flow, and data transformations of the implemented parallel-beam X-ray motion simulator.

---

## 1. Mathematical & Physical Model

The simulator is built on two primary computations: spatial attenuation via orthographic projection and temporal integration via stochastic sampling.

### 1.1 Spatial Attenuation (Beer-Lambert Law)
The transmission of X-rays through the voxel volume is modeled using the Beer-Lambert law. For a parallel beam, the intensity of the transmitted beam $I$ is calculated as the exponential decay of the line integral of linear attenuation coefficients $\mu$:

$$I = \exp\left(-\int \mu(s) ds\right)$$

In this implementation:
* The beam is perfectly parallel (orthographic), meaning there is zero geometric magnification and no beam divergence.
* The line integral is computed as a discrete summation along the specified array axis.
* Because straight-line summation is commutative, beam direction is mathematically irrelevant. **AP (Antero-Posterior) and PA (Postero-Anterior) views yield identical results.**

### 1.2 Temporal Integration and Motion Blur
To simulate the exposure window of an X-ray detector, the system calculates the time-average of instantaneous projections over the total `exposure_time` ($T$):

$$I_{det} = \frac{1}{T} \int_0^T I(t) dt$$

### 1.3 Stratified Monte Carlo Sampling
The temporal integral is approximated by dividing $T$ into `n_steps` intervals of duration $dt$. A random time sample $t_i$ is drawn from each sub-interval:

$$t_i = (i + U[0,1]) \cdot dt \quad \text{for} \quad i = 0 \dots N-1$$

This stratified approach breaks the degeneracy of fixed-midpoint sampling. It correctly simulates how low-dose or fast-moving structures produce grainy, aliased blur, while high-sampling converges to smooth motion blur, mimicking the physical reality of Poisson-distributed photon arrivals.

---

## 2. Configuration & Parameters

The simulation state is governed by the `SimulationParams` dataclass and a predefined axis mapping.

### 2.1 State Definition (`SimulationParams`)
```python
@dataclass(frozen=True)
class SimulationParams:
    body_part: str
    proj_axis: int        # Driven by PROJ_AXES mapping
    exposure_time: float  # Total integration time T
    n_photons: int
    motion_type: str      # 'none', 'linear', or 'breathing'
    velocity: float
    amplitude: float
    frequency: float
    motion_axis: int      # Axis of physical displacement
    n_steps: int          # Integration resolution (N)
    noise_type: str
    mitigation: str
```

### 2.2 View Mapping (`PROJ_AXES`)
Physical clinical views are mapped to NumPy array axes. Due to the parallel beam assumption, opposing views share the same projection axis:
```python
PROJ_AXES = {
    "AP  (Front -> Back)": 1,
    "PA  (Back -> Front)": 1,
    "Lateral  (Left -> Right)": 0,
    "Lateral  (Right -> Left)": 0,
}
```

---

## 3. Data Pipeline

The system transforms a static 3D array of attenuation coefficients into a dynamic 2D transmission map.

| Stage | Input Data | Transformation | Output Data |
| :--- | :--- | :--- | :--- |
| **1. Initialization** | 3D Volume, `body_part` | Anatomical mask generation ($w_z$). | 3D Array + 1D Mask Vector |
| **2. Kinematics** | `motion_type`, Time $t$ | Calculate spatial displacement $d_{cm}$. | Float (Shift in voxels) |
| **3. Spatial Shift**| 3D Volume, shift, $w_z$ | `ndimage.shift` or masked translation. | Shifted 3D Volume |
| **4. Orthographic Summation** | Shifted 3D Volume, `proj_axis` | Matrix sum along axis $\times$ `VOXEL_SIZE`. | 2D Array (Optical Depth) |
| **5. Attenuation** | 2D Array (Optical Depth) | Apply exponentiation $e^{-x}$. | 2D Array (Transmission Frame) |
| **6. Accumulation** | `n_steps` $\times$ 2D Frames | Element-wise addition and division by $N$.| **Final 2D Image (float32)** |

---

## 4. Core Code Logic

### 4.1 Orthographic Projection
The purely spatial calculation compressing the 3D volume into a 2D transmission map.
```python
# Matrix summation along the projection axis (equivalent to parallel line integrals)
line_integral = np.sum(volume, axis=axis, dtype=np.float64) * VOXEL_SIZE

# Exponentiate negative integral to calculate final transmission intensity
proj = np.exp(-line_integral).astype(np.float32)
```

### 4.2 Temporal Motion Loop
The generation of time samples, application of spatial shifts, and accumulation of projection frames.

```python
dt = p.exposure_time / p.n_steps

for i in range(p.n_steps):
    # Stratified Monte Carlo time sample
    t = (i + np.random.uniform(0.0, 1.0)) * dt

    # Kinematic displacement
    if motion_type == "breathing":
        d_cm = p.amplitude * np.sin(2.0 * np.pi * p.frequency * t)

    # Apply physical shift in voxels (using anatomical mask w_z)
    d_vox = d_cm / VOXEL_SIZE
    moved = _shift_with_mask(volume, d_vox, p.motion_axis, w_z)

    # Project frame and accumulate
    proj  = project_parallel(moved, p.proj_axis)
    accum = proj if accum is None else accum + proj

# Time-averaged output
final_image = (accum / p.n_steps).astype(np.float32)
```