***

## 🧮 Mathematical Details

### 1. X-Ray Projection (Beer-Lambert Law)

The transmission of X-rays through the body is modeled using the Beer-Lambert law. The intensity $I(u,v)$ detected at pixel $(u,v)$ is given by:

$$
I(u,v) = I_0 \cdot \exp\left(- \int \mu(x,y,z)\, dl \right)
$$

Where:

- $I_0$ is the incident beam intensity (normalized to $1.0$)  
- $\mu(x,y,z)$ is the local linear attenuation coefficient
- $dl$ is the differential path length  

Discrete form used in the software:

$$
I_{\text{discrete}}(u,v) = \exp\left(- \sum_i \mu_i(u,v) \cdot \Delta l \right)
$$

Where $\Delta l = 0.30$ cm (VOXEL\_SIZE).

---

### 2. Motion Artifacts (Temporal Integration)

The final detected signal is the time-average over exposure window $T$:

$$
P_{\text{final}}(u,v) = \frac{1}{T} \int_0^T P(u,v; t)\, dt \approx \frac{1}{N} \sum_{i=0}^{N-1} P(u,v; t_i)
$$

#### Displacement Models $d(t)$

- **Linear Motion:**

$$
d(t) = v \cdot t
$$

- **Sinusoidal Motion:**

$$
d(t) = A \cdot \sin(2\pi f t)
$$

---

### 3. Artifact Mitigation Techniques

#### Unsharp Masking

$$
I_{\text{mitigated}} = I + 0.65 \cdot \left( I - (I * G_{\sigma}) \right)
$$

Where $G_{\sigma}$ is a Gaussian kernel with $\sigma = 1.5$.

#### Richardson-Lucy Deconvolution

$$
u^{(k+1)} = u^{(k)} \cdot \left( \tilde{h} * \frac{d}{h * u^{(k)}} \right)
$$

Where:

- $\tilde{h}$ is the flipped PSF  
- $d$ is the observed blurry image  
- denotes convolution  

---

### 4. Evaluation Metrics

#### Normalized Mean Square Error (NMSE)

Calculates the error between the noisy image and the reference image, normalized by the energy of the reference image:

$$
\text{NMSE} = \frac{|| I_{\text{ref}} - I_{\text{noisy}} ||^2}{|| I_{\text{ref}} ||^2} = \frac{\sum \left( I_{\text{ref}}(u,v) - I_{\text{noisy}}(u,v) \right)^2}{\sum I_{\text{ref}}(u,v)^2}
$$

#### Structural Similarity Index Measure (SSIM)

Measures the perceptual similarity between two images utilizing local uniform windows (size $7 \times 7$):

$$
\text{SSIM} = \frac{(2\mu_{\text{ref}}\mu_{\text{noisy}} + c_1)(2\sigma_{\text{ref},\text{noisy}} + c_2)}{(\mu_{\text{ref}}^2 + \mu_{\text{noisy}}^2 + c_1)(\sigma_{\text{ref}}^2 + \sigma_{\text{noisy}}^2 + c_2)}
$$

Where:

- $\mu$ represents the local mean of the respective image  
- $\sigma^2$ represents the local variance  
- $\sigma_{\text{ref},\text{noisy}}$ represents the local covariance between the reference and noisy images  
- $c_1 = (0.01 \cdot L)^2$ and $c_2 = (0.03 \cdot L)^2$ act as stabilization constants, with the dynamic range $L = 1.0$
