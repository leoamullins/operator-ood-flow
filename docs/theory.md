# Theory

[← Back to README](../README.md) · [Methodology](methodology.md) · [Findings](findings.md) · [Issues log](issues-log.md)

The mathematical background behind the solver, the shape families, the model and the metrics, with the assumptions each one relies on.

---

## 1. The operator being learned

For an obstacle $K \subset \mathbb{R}^2$ with signed distance function $z = \mathrm{sdf}_K$, steady incompressible flow at Reynolds number $\mathrm{Re}$ defines the solution operator

$$
\mathcal{G}: z \mapsto q = (u_x, u_y, p) \quad \text{on the fluid domain } \Omega \setminus K,
$$

where $q$ solves the steady Navier–Stokes equations with uniform inflow $u_\mathrm{in}$ and no slip on $\partial K$. The DeepONet $\mathcal{G}_\theta$ approximates $\mathcal{G}$ from samples $\{(z_i, q_i)\}$ in which every $K_i$ is smooth and near-convex. *Out of distribution* means evaluating $\mathcal{G}_\theta(z)$ for a $K$ outside that family.

**Assumptions:** 2D and steady: Re = 20 is well below the vortex-shedding onset for a circular cylinder (Re ≈ 47), and the solver's convergence criterion confirms every shape reached a steady state, a single Re and body size, and $\mathcal{G}$ well defined for every shape tested.

---

## 2. Lattice Boltzmann solver

**BGK dynamics.** On the D2Q9 lattice with velocities $c_i$ and weights $w_i$ (4/9, 1/9 × 4, 1/36 × 4), the populations evolve as

$$
f_i(x + c_i, t + 1) = f_i(x, t) - \frac{1}{\tau}\left(f_i - f_i^{\mathrm{eq}}\right),
\qquad
f_i^{\mathrm{eq}} = w_i\rho\left(1 + 3\,c_i\!\cdot u + \tfrac{9}{2}(c_i\!\cdot u)^2 - \tfrac{3}{2}|u|^2\right),
$$

with $\rho = \sum_i f_i$ and $\rho u = \sum_i c_i f_i$.

**Viscosity and Reynolds number.** The Chapman–Enskog expansion recovers the Navier–Stokes equations with $c_s^2 = 1/3$, pressure $p = c_s^2\rho$ and kinematic viscosity $\nu = c_s^2(\tau - \tfrac12)$. Setting $\mathrm{Re} = u_\mathrm{in}D/\nu$ with $D = 2r_0$ gives

$$
\tau = \frac{3\,u_\mathrm{in}D}{\mathrm{Re}} + \frac12 ,
$$

which is 1.1 at s = 2 ($D$ = 40 cells, $u_\mathrm{in}$ = 0.1, Re = 20).

**Error sources:**
- **Compressibility:** the scheme reproduces incompressible flow up to $O(\mathrm{Ma}^2)$ errors, with $\mathrm{Ma} = u_\mathrm{in}/c_s \approx 0.17$.
- **Walls:** bounce-back on a staircased boundary is accurate to only first order for curved and oblique walls, and least accurate at sharp vertices. This fits polygons converging slowest in the grid study ([issues log #5](issues-log.md)).
- **Convergence:** the flow counts as steady when the maximum velocity change over 2000 steps is below $10^{-5}$.

---

## 3. Shape families

### Periodic Gaussian random functions

The boundary perturbation $g(\theta)$ is a zero-mean Gaussian process on $[0, 2\pi)$ with the periodic squared-exponential kernel

$$
k(\theta, \theta') = \exp\!\left(-\frac{2\sin^2\!\big((\theta - \theta')/2\big)}{\ell^2}\right).
$$

Samples are drawn as $g = Lz$, where $K = LL^\top$ is the Cholesky factorisation and $z \sim N(0, I)$.

### Convexity margin

For a polar curve $r(\theta)$, the signed curvature is

$$
\kappa(\theta) = \frac{r^2 + 2r'^2 - r\,r''}{\left(r^2 + r'^2\right)^{3/2}}.
$$

The denominator is positive, so the shape is convex exactly when the numerator is non-negative everywhere. The code uses the dimensionless margin

$$
m = \min_\theta \frac{r^2 + 2r'^2 - r\,r''}{r^2},
$$

which is 1 for a circle and negative where the boundary is concave. It is a scaled version of the curvature numerator, not the curvature itself. $r'$ and $r''$ are computed spectrally with the FFT, dropping the Nyquist mode for the odd derivative.

### Equal-area lobes

For $r = r_0(1 + a\cos k\theta)/\sqrt{1 + a^2/2}$:

$$
A = \tfrac12\int_0^{2\pi} r^2\,d\theta = \frac{r_0^2}{2}\cdot\frac{2\pi + \pi a^2}{1 + a^2/2} = \pi r_0^2 ,
$$

independent of $k$ and $a$.

### Equal-area rounded polygons

A regular n-gon with circumradius $R$ has area $A(R) = \tfrac{n}{2}R^2\sin\tfrac{2\pi}{n}$ and perimeter $P(R) = 2nR\sin\tfrac{\pi}{n}$. Growing it outward by a distance $\rho$ rounds every corner with radius $\rho$. By Steiner's formula for convex sets,

$$
A_\rho = A(R) + P(R)\,\rho + \pi\rho^2 .
$$

Setting $A_\rho = \pi r_0^2$ gives a quadratic in $R$, and its positive root fixes the inner polygon. The rounding centres are the inner polygon's vertices; the sharp shape ($\rho = 0$) has $R_0 = r_0\sqrt{2\pi / (n\sin(2\pi/n))}$.

---

## 4. DeepONet

$$
\mathcal{G}_\theta(z)(y)_j = \sum_{k=1}^{p} b_{jk}(z)\, t_{jk}(y) + b_{0j}, \qquad j \in \{u_x, u_y, p\},
$$

with the branch net $b$ acting on the downsampled SDF and the trunk net $t$ acting on the query point $y$. The universal approximation theorem for operators (Chen & Chen 1995; Lu et al. 2021) guarantees that this form can approximate any continuous operator uniformly on a *compact* set of inputs to any accuracy. That guarantee says nothing about inputs outside the compact set the model was trained on, which is the setting studied here.

---

## 5. Error metrics

All norms are discrete sums over a region $\Omega'$ of lattice cells. Every cell has the same area, so the cell area cancels in any ratio, and each sum approximates the continuous $L^2(\Omega')$ norm.

- **Relative $L^2$ per field** (`ux`, `uy`, `p` and the `_wall` versions):
$$
e_j = \frac{\|\hat q_j - q_j\|_{\Omega'}}{\|q_j\|_{\Omega'}} .
$$
- **Velocity vector** (`vel`, `vel_wall`): the same, with $|v|^2 = v_x^2 + v_y^2$ summed over both components. This is *not* a combination of $e_{u_x}$ and $e_{u_y}$: two ratios can't be added back together without their numerators and denominators ([issues log #9](issues-log.md)). It always lies between the two per-component values.
- **Near-wall RMSE** (`vel_corner_rmse`):
$$
\mathrm{RMSE}/u_\mathrm{in} = \frac{1}{u_\mathrm{in}}\sqrt{\frac{1}{|\Omega'|}\sum_{\Omega'} |\hat u - u|^2}.
$$
No slip forces $u \to 0$ at the wall, which makes the denominator of a relative error small near the wall. An absolute error scaled by $u_\mathrm{in}$ avoids that inflation ([issues log #10](issues-log.md)).
- **Periodic 1D norms** (`metrics.relative_velocity_error`): on a uniform grid with $N$ points over one period, the trapezoid rule $\int_0^{2\pi} f^2\,d\theta \approx \Delta\theta\sum_j f_j^2$ is spectrally accurate for smooth periodic $f$. $\Delta\theta$ cancels in the ratio, which reduces to `norm(pred - true) / norm(true)`. The endpoint must not be repeated ([issues log #8](issues-log.md)).
- **SDF distance** (`sdf_native`, `sdf_branch`): $\|z_\rho - z_0\|_2$ multiplied by $\sqrt{\text{pixel area}}$ (in $r_0^2$), so both resolutions approximate the same continuous norm and can be compared.

---

## 6. Response error identity

Let $\Delta q = q_\rho - q_0$ be the change in the true flow and $\Delta\hat q = \hat q_\rho - \hat q_0$ the change in the prediction, both on the same cells with the same inner product. Define

$$
R = \frac{\|\Delta\hat q\|}{\|\Delta q\|}, \qquad
C = \frac{\langle \Delta q, \Delta\hat q\rangle}{\|\Delta q\|\,\|\Delta\hat q\|}.
$$

Expanding the square,

$$
\|\Delta\hat q - \Delta q\|^2 = \|\Delta\hat q\|^2 - 2\langle\Delta q, \Delta\hat q\rangle + \|\Delta q\|^2 ,
$$

and dividing by $\|\Delta q\|^2$ gives

$$
E = \frac{\|\Delta\hat q - \Delta q\|}{\|\Delta q\|} = \sqrt{1 + R^2 - 2RC}.
$$

Consequences:

- **No response scores 1:** $E(R = 0) = 1$, whatever $C$ is.
- **The response along the true change** is $\langle \Delta\hat q, \Delta q\rangle / \|\Delta q\|^2 = RC$. It is 1 for a perfect response.
- **Alignment limits what scaling can achieve.** For a fixed $C$, $E$ is smallest at $R = C$, where $E_{\min} = \sqrt{1 - C^2}$. With the measured $C \approx 0.5$, even a perfectly scaled response would leave $E \approx 0.87$. Getting the size right alone wouldn't fix the response; the direction is also only partly right.
- **Averaging caveat:** the identity holds per shape. The mean of per-shape $E$ differs from $\sqrt{1 + \bar R^2 - 2\bar R\bar C}$.

**Assumptions:** both changes are measured on the same cells (the corner region is fixed across ρ for exactly this reason), with the same weights and the same norm over the output components.
