## 🧮 Mathematical Details

### 1. X-Ray Projection (Beer-Lambert Law)

The transmission of X-rays through the body is modeled using the Beer-Lambert law. The intensity $I(u,v)$ detected at pixel $(u,v)$ is given by:

$$
I(u,v) = I_0 \cdot \exp\left(- \int \mu(x,y,z)\, dl \right)
$$

Where:

- $I_0$ is the incident beam intensity (normalized to $1.0$)  
- $\mu(x,y,z)$ is the local linear attenuation coefficient (in cm$^{-1}$)  
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

### 3. Detector Noise Models

Let $I$ be the noise-free intensity ($0 \le I \le 1$), and $N_0$ be the incident photon flux.

#### Poisson Noise

$$
k \sim \text{Poisson}(N_0 \cdot I)
$$

$$
\hat{I} = \frac{k}{N_0}
$$

#### Gaussian Noise

$$
\hat{I} = I + \mathcal{N}(0, \sigma^2), \quad \sigma = \frac{1}{\sqrt{N_0}}
$$

#### Combined Model

$$
k \sim \text{Poisson}(N_0 \cdot I)
$$

$$
r \sim \mathcal{N}(0, \sigma_{\text{read}}^2), \quad \sigma_{\text{read}} = 0.01 \cdot \sqrt{N_0}
$$

$$
\hat{I} = \frac{k + r}{N_0}
$$

---

### 4. Artifact Mitigation Techniques

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
- $*$ denotes convolution  

---

### 5. Evaluation Metrics

#### Signal-to-Noise Ratio (SNR)

$$
\text{SNR (dB)} = 20 \cdot \log_{10}\left( \frac{\text{RMS}_{\text{signal}}}{\text{RMS}_{\text{noise}}} \right)
$$

#### Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{M} \sum \left( I(u,v) - K(u,v) \right)^2
$$

#### Peak Signal-to-Noise Ratio (PSNR)

$$
\text{PSNR (dB)} = 10 \cdot \log_{10}\left( \frac{MAX_I^2}{\text{MSE}} \right)
$$

Where $MAX_I = 1.0$.
