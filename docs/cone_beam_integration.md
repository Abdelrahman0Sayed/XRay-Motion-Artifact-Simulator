# Cone-Beam Integration Notes

This document explains the new cone-beam calculations and how they were integrated into the existing parallel projection pipeline.

## 1) New parameters added to the simulation payload

The cone-beam model needs geometry and coordinate-system context that the parallel model does not. The payload now includes:

- projection_type: "parallel" or "cone" to select the projection path.
- view_dir: One of "AP", "PA", "LR", "RL". Used to orient the source relative to the phantom.
- sod: Source-to-object-center distance in cm.
- sdd: Source-to-detector distance in cm.
- voxel_size: Voxel size in cm (used to convert index steps to physical distance).
- phantom_shape: Full phantom shape (nx, ny, nz) so body-part crops can be aligned to the phantom coordinate system.

These values let the cone-beam code compute per-slice distances to the source using phantom coordinates rather than assuming a cropped volume starts at zero depth.

Code sketch:

```python
params = SimulationParams(
   body_part=body_part,
   proj_axis=proj_axis,
   projection_type=projection_type,
   view_dir=view_dir,
   sod=sod,
   sdd=sdd,
   voxel_size=VOXEL_SIZE,
   phantom_shape=phantom_shape,
   exposure_time=exposure_time,
   n_photons=n_photons,
   motion_type=motion_type,
   velocity=velocity,
   amplitude=amplitude,
   frequency=frequency,
   motion_axis=motion_axis,
   n_steps=n_steps,
   noise_type=noise_type,
   mitigation=mitigation,
)
```

## 2) Cone-beam distance model (per-slice SOD)

The cone-beam approximation treats each slice as a planar object at a distinct distance from the source, which defines its magnification. The new logic computes the physical position of each slice along the projection axis in centimeters, then adds it to the SOD to get per-slice source distance.

### 2.1 Slice positions for axial (Z) projections

For Z-axis projection (proj_axis == 2), slice positions are computed from the phantom coordinate system. This is important because the crop depends on body_part, so a slice index in a cropped sub-volume does not directly correspond to the phantom centerline.

Steps:

1) Determine the phantom span in cm on the projection axis:
   full_span_cm = (full_n - 1) * voxel_size
   where full_n comes from phantom_shape for the chosen axis.

2) Map the body_part Z-range from BODY_PARTS (normalized in [-1, 1]) to cm:
   slice_pos_cm = z_norm * (full_span_cm / 2)
   where z_norm is linearly spaced between the body_part z_low and z_high.

This produces a slice position in the phantom coordinate system, centered at 0 cm (phantom center).

Code sketch:

```python
full_n = phantom_shape[proj_axis]
full_span_cm = max(1, full_n - 1) * voxel_size
z_lo, z_hi = BODY_PARTS.get(body_part, (-1.0, 1.0))
z_norm = np.linspace(z_lo, z_hi, n_slices)
slice_pos_cm = z_norm * (full_span_cm / 2.0)
```

### 2.2 Slice positions for lateral (X/Y) projections

For lateral projections (proj_axis 0 or 1), there is no body_part-specific coordinate range in the current model. Positions are approximated as uniform offsets around the center of the cropped volume:

- center = (n_slices - 1) / 2
- slice_pos_cm = (index - center) * voxel_size

This preserves a symmetric, center-aligned geometry for lateral cone-beam views.

Code sketch:

```python
center = (n_slices - 1) / 2.0
slice_pos_cm = (np.arange(n_slices, dtype=np.float64) - center) * voxel_size
```

### 2.3 Applying view direction

View direction flips the sign of slice positions for PA or RL views:

- reverse_dirs = {"PA", "RL"}
- if view_dir in reverse_dirs: slice_pos_cm = -slice_pos_cm

This means AP/LR treat positive offsets as moving away from the source, while PA/RL mirror the geometry.

Code sketch:

```python
reverse_dirs = {"PA", "RL"}
if view_dir in reverse_dirs:
   slice_pos_cm = -slice_pos_cm
```

### 2.4 Per-slice distance to source

Per-slice source distance is computed as:

- d_slice = sod + slice_pos_cm

Then it is clamped to a small positive value to avoid division by zero or negative distances:

- d_slice = clip(d_slice, voxel_size, +inf)

This is used for the slice magnification and for detector sizing.

Code sketch:

```python
d_slices = sod + slice_pos_cm
d_slices = np.clip(d_slices, voxel_size, None)
```

## 3) Cone-beam magnification and accumulation

Magnification for each slice is:

- mag = sdd / d_slice

Each slice is scaled by its magnification before being added into the detector plane. The scaled slice is centered onto the detector grid, and its contribution is added to the Beer-Lambert line integral:

- line_integral += scaled_slice * voxel_size * mag

The extra factor of mag accounts for the longer path length through the scaled slice when projected onto the detector plane.

Code sketch:

```python
mag = sdd / d_slice
scaled_slice = ndimage.zoom(slc, mag, order=1, mode="constant", cval=0.0)
line_integral[uu:uu + scaled_slice.shape[0], vv:vv + scaled_slice.shape[1]] += (
   scaled_slice * voxel_size * mag
)
```

## 4) Robust detector sizing

The detector plane is sized to fit the maximum expected magnification. The size is computed using ceil (not floor) to prevent off-by-one truncation:

- det_shape = ceil(base_shape * max_mag)

If a scaled slice still exceeds the detector bounds due to rounding or edge-case geometry, the detector accumulator is expanded dynamically using centered padding. This prevents shape mismatches and runtime errors during accumulation.

Code sketch:

```python
det_shape = (
   max(1, int(np.ceil(base_shape[0] * max_mag))),
   max(1, int(np.ceil(base_shape[1] * max_mag))),
)

if (scaled_slice.shape[0] > det_shape[0]) or (scaled_slice.shape[1] > det_shape[1]):
   new_shape = (
      max(det_shape[0], scaled_slice.shape[0]),
      max(det_shape[1], scaled_slice.shape[1]),
   )
   pad_u = (new_shape[0] - det_shape[0]) // 2
   pad_v = (new_shape[1] - det_shape[1]) // 2
   line_integral = np.pad(
      line_integral,
      ((pad_u, new_shape[0] - det_shape[0] - pad_u),
       (pad_v, new_shape[1] - det_shape[1] - pad_v)),
      mode="constant",
   )
   det_shape = line_integral.shape
```

## 5) Integration into the simulation pipeline

- The UI now provides a projection model selector (Parallel or Cone Beam).
- SOD and SDD are enabled only for Cone Beam to avoid accidental misuse.
- The simulation worker reads projection_type and chooses the correct projection path.
- The cone-beam projection uses the new parameters and phantom coordinate alignment to compute per-slice magnification.

Code sketch:

```python
if projection_type == "parallel":
   static_proj = project_parallel(vol, p.proj_axis)
   motion_proj = simulate_parallel_acquisition(vol, p)
elif projection_type == "cone":
   static_proj = project_cone_beam(vol, p)
   motion_proj = simulate_cone_acquisition(vol, p)
```

## 6) Why this matters visually

The phantom view shows a source and detector marker. The cone-beam calculations now match the phantom coordinate system so that:

- Changing SOD moves the effective source distance relative to the phantom center.
- Changing SDD changes magnification as expected.
- Cropped body parts are still aligned to their true Z positions within the full phantom.

This keeps the cone-beam projection consistent with the visual model in the GUI.
