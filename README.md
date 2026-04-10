# X-Ray Motion Artifact Simulator

This software is an educational simulator for Medical Imaging, specifically designed to demonstrate the effects of patient motion during X-ray acquisition and evaluate various mitigation strategies. 

## 🔄 Program Flow

The simulator operates in a pipeline that mimics a real-world X-ray acquisition process:

1. **Phantom Generation**: On startup, a 3D numerical human phantom is created. It is composed of overlapping axis-aligned ellipsoids representing different anatomical structures (bones, lungs, soft tissue, etc.), each assigned realistic linear attenuation coefficients.
2. **User Configuration**: The user configures the imaging target (e.g., Chest, Head), viewing direction (AP/PA/Lateral), motion type/parameters, exposure settings, noise models, and post-processing mitigations.
3. **Simulation Loop (Temporal Integration)**: 
   - The exposure time is discretized into $N$ sub-intervals.
   - For each time step $t_i$, the programmed motion model calculates a displacement vector.
   - The 3D phantom is subjected to a rigid-body translation using a fast nearest-neighbor interpolation.
   - A 2D projection of the shifted phantom is captured according to the Beer-Lambert law.
   - Projected images from all time steps are averaged to form the final motion-blurred "clean" image.
4. **Noise Addition**: The selected detector noise (Poisson, Gaussian, or Combined) is applied to the averaged projection, incorporating photon flux parameters.
5. **Mitigation**: Post-acquisition restoration techniques (e.g., filtering, deconvolution) are applied to the noisy, blurred image.
6. **Metrics & Display**: Signal-to-Noise Ratio (SNR) and Peak SNR (PSNR) are calculated. The application updates the GUI with three views: the ideal static image, the motion-artifact image, and the mitigated image.

---

## 🧮 Mathematical Details

### 1. X-Ray Projection (Beer-Lambert Law)
The transmission of X-rays through the body is modeled using the Beer-Lambert law. The intensity $I(u,v)$ detected at pixel $(u,v)$ is given by:

$$ I(u,v) = I_0 \cdot \exp\left( - \int \mu(x,y,z) \, dl \right) $$

Where:
- $I_0$ is the incident beam intensity (normalized to $1.0$).
- $\mu(x,y,z)$ is the local linear attenuation coefficient (in cm⁻¹) of the tissue at 70 keV.
- $dl$ is the differential path length along the projection axis.

In discrete terms used by the software:
$$ I_{discrete}(u,v) = \exp\left( - \sum_{i} \mu_i(u,v) \cdot \Delta l \right) $$
Where $\Delta l$ is the `VOXEL_SIZE` ($0.30$ cm).

### 2. Motion Artifacts (Temporal Integration)
Because patient motion occurs *during* the exposure, the final detected signal is the time-average of instantaneous projections over the exposure window $T$:

$$ P_{final}(u,v) = \frac{1}{T} \int_0^T P(u,v; t) \, dt \approx \frac{1}{N} \sum_{i=0}^{N-1} P(u,v; t_i) $$

**Displacement Models $d(t)$**:
- **Linear Motion**: Represents uniform drift.
  $$ d(t) = v \cdot t $$
- **Sinusoidal Motion**: Represents periodic movements like breathing or heartbeats.
  $$ d(t) = A \cdot \sin(2\pi f t) $$
The calculated metric $d(t)$ is converted to voxel shifts and applied to the 3D volume before computing $P(u,v; t_i)$.

### 3. Detector Noise Models
Let $I$ be the noise-free normalized intensity ($0 \le I \le 1$), and $N_0$ be the incident photon flux per detector element.

- **Poisson (Quantum/Photon-counting Noise)**:
  The number of detected photons $k$ follows a Poisson distribution:
  $$ k \sim \text{Poisson}(N_0 \cdot I) $$
  The normalized noisy image is $\hat{I} = k / N_0$.
- **Gaussian (Electronic Read-out Noise)**:
  Modeled as normally distributed noise added to the signal:
  $$ \hat{I} = I + \mathcal{N}(0, \sigma^2) \quad \text{where} \quad \sigma = \frac{1}{\sqrt{N_0}} $$
- **Combined Model**:
  Adds severe baseline electronic read-out noise (roughly $1\%$ of full scale) to the quantum noise:
  $$ k \sim \text{Poisson}(N_0 \cdot I) $$
  $$ r \sim \mathcal{N}(0, \sigma_{read}^2) \quad \text{where} \quad \sigma_{read} = 0.01 \cdot \sqrt{N_0} $$
  $$ \hat{I} = \frac{k + r}{N_0} $$

### 4. Artifact Mitigation Techniques

- **Unsharp Masking**:
  Enhances edges by subtracting a blurred version of the image.
  $$ I_{mitigated} = I + 0.65 \cdot (I - (I * G_{\sigma})) $$
  Where $G_{\sigma}$ is a Gaussian kernel with $\sigma=1.5$.
  
- **Richardson-Lucy (RL) Deconvolution**:
  An iterative probabilistic method to solve for the unblurred image $u$, assuming a known Point Spread Function (PSF) $h$. The approximation uses a uniform box kernel modeling linear blur:
  $$ u^{(k+1)} = u^{(k)} \cdot \left( \tilde{h} * \frac{d}{h * u^{(k)}} \right) $$
  Where $\tilde{h}$ is the flipped PSF, $d$ is the observed blurry image, and $*$ denotes convolution.

### 5. Evaluation Metrics

- **Signal-to-Noise Ratio (SNR)**:
  Evaluates signal power relative to noise power:
  $$ \text{SNR (dB)} = 20 \cdot \log_{10}\left( \frac{\text{RMS}_{signal}}{\text{RMS}_{noise}} \right) $$
  
- **Peak Signal-to-Noise Ratio (PSNR)**:
  Uses Mean Squared Error (MSE) between the reference (ideal static) image $I$ and the test image $K$:
  $$ \text{MSE} = \frac{1}{M} \sum \left( I(u,v) - K(u,v) \right)^2 $$
  $$ \text{PSNR (dB)} = 10 \cdot \log_{10}\left( \frac{MAX_I^2}{\text{MSE}} \right) $$
  Where $MAX_I = 1.0$ (since transmission signals are normalized).