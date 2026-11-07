from itertools import product

import numpy as np
from scipy.ndimage import distance_transform_edt

from .grf import build_cholesky_factor, sample_functions

N_THETA = 128
THETA = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)


def sample_radii(n_shapes, r0, eps, ell, rng, a1_max=0.3, a2_max=0.25):
    """Draw polar radii on THETA using two harmonics and periodic Gaussian noise."""
    L = build_cholesky_factor(THETA, ell, periodic=True)
    g = sample_functions(L, n_shapes, rng)
    a1 = rng.uniform(0, a1_max, size=(n_shapes, 1))
    a2 = rng.uniform(0, a2_max, size=(n_shapes, 1))
    phi1 = rng.uniform(0, 2 * np.pi, size=(n_shapes, 1))
    phi2 = rng.uniform(0, 2 * np.pi, size=(n_shapes, 1))
    radii = r0 * (
        1 + a1 * np.cos(THETA - phi1) + a2 * np.cos(2 * (THETA - phi2)) + eps * g
    )
    if np.any(radii <= 0):
        raise ValueError("Non-positive radius: reduce eps")
    return radii


def shape_mask(radii, Ny, Nx, cx, cy):
    """Rasterise one polar boundary onto a lattice. True marks solid cells."""
    y, x = np.mgrid[0:Ny, 0:Nx]
    dx, dy = x - cx, y - cy
    angle = np.arctan2(dy, dx) % (2 * np.pi)  # map to [0, 2π)
    dist = np.hypot(dx, dy)

    r_boundary = np.interp(angle, THETA, radii, period=2 * np.pi)
    return dist <= r_boundary


def signed_distance(mask):
    """Return distances in lattice cells, negative inside the solid."""
    return distance_transform_edt(~mask) - distance_transform_edt(mask)


def sample_convex_shapes(
    n_shapes,
    Ny,
    Nx,
    cx,
    cy,
    r0,
    eps,
    ell,
    rng,
    a1_max=0.3,
    a2_max=0.18,
    tol=-0.2,
    max_rounds=100,
):
    """Return masks, radii, and margins for samples with margin at least tol."""
    kept_radii, kept_margins = [], []
    for _ in range(max_rounds):
        if len(kept_radii) >= n_shapes:
            break
        radii = sample_radii(n_shapes, r0, eps, ell, rng, a1_max=a1_max, a2_max=a2_max)
        convex, margin = is_convex(radii, tol=tol)
        kept_radii.extend(radii[convex])
        kept_margins.extend(margin[convex])
    if len(kept_radii) < n_shapes:
        raise RuntimeError(
            f"only {len(kept_radii)}/{n_shapes} convex shapes after {max_rounds} rounds; loosen tol or amplitudes"
        )

    radii = np.array(kept_radii[:n_shapes])
    margins = np.array(kept_margins[:n_shapes])
    masks = np.array([shape_mask(r, Ny, Nx, cx, cy) for r in radii])
    return masks, radii, margins


def sample_lobed_shapes(
    n_shapes, Ny, Nx, cx, cy, r0, rng, amp=(0.1, 0.4), lobes=(3, 4, 5)
):
    """Draw r = r0 (1 + a cos k(theta - phi)) and return masks, radii, and margins."""
    k = rng.choice(lobes, size=(n_shapes, 1))
    a = rng.uniform(*amp, size=(n_shapes, 1))
    phi = rng.uniform(0, 2 * np.pi, size=(n_shapes, 1))
    radii = r0 * (1 + a * np.cos(k * (THETA - phi)))
    _, margins = is_convex(radii)
    masks = np.array([shape_mask(r, Ny, Nx, cx, cy) for r in radii])
    return masks, radii, margins


def is_convex(radii, tol=0.0):
    """Use FFT derivatives to compute the convexity margin and test it against tol."""
    N = radii.shape[-1]
    k = np.fft.fftfreq(N, d=1 / N)  # integer wavenumbers, since theta has period 2pi
    r_hat = np.fft.fft(radii, axis=-1)

    k1 = k.copy()
    k1[N // 2] = 0  # drop the Nyquist mode for the odd derivative
    r1 = np.fft.ifft(1j * k1 * r_hat, axis=-1).real  # dr/dtheta
    r2 = np.fft.ifft(-(k**2) * r_hat, axis=-1).real  # d2r/dtheta2

    curvature_numerator = radii**2 + 2 * r1**2 - radii * r2
    margin = (curvature_numerator / radii**2).min(axis=-1)  # dimensionless; 1.0 for a perfect circle
    return margin >= tol, margin


# -------
# SAMPLING POLYGONS WITH CORNERS


def sample_polygon_shapes(n_shapes, Ny, Nx, cx, cy, r0, rng, sides=(3, 4, 5, 6, 7, 8)):
    ns = sides

    y, x = np.mgrid[0:Ny, 0:Nx]
    px, py = x - cx, y - cy

    masks = []
    sides_used = []
    radii = []
    for _ in range(n_shapes):
        n = rng.choice(ns)
        phi = rng.uniform(0, 2 * np.pi)

        sides_used.append(n)

        # choose R so that every polygon has same area as a circle of radius r0
        R = r0 * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))

        corners = []
        for j in range(n):
            # corners sit at (Rcos, Rsin): generate a tuple coordinate
            corners.append(
                (
                    R * np.cos(phi + 2 * np.pi * j / n),
                    R * np.sin(phi + 2 * np.pi * j / n),
                )
            )

        inside = np.ones((Ny, Nx), dtype=bool)
        for j in range(n):
            sx, sy = corners[j]  # start of an edge
            tx, ty = corners[(j + 1) % n]  # end of edge, wraps back to 0
            ex, ey = tx - sx, ty - sy  # edge vector
            qx, qy = px - sx, py - sy  # start corner - > every grid point
            inside &= (ex * qy - ey * qx) >= 0

        masks.append(inside)
        radii.append(
            R * np.cos(np.pi / n) / np.cos((THETA - phi) % (2 * np.pi / n) - np.pi / n)
        )

    return (
        np.array(masks),
        np.array(radii),
        np.zeros(n_shapes),
        {"sides": np.array(sides_used)},
    )


# -------
# SYSTEMATIC GRIDS: every shape set by its parameters, no random draws


def sample_lobed_grid(
    n_shapes, Ny, Nx, cx, cy, r0, rng,
    ks=(3, 4, 5), amps=(0.04, 0.08, 0.12, 0.16, 0.2), n_orient=12,
):
    """Equal-area lobes r = r0 (1 + a cos k(theta - phi)) / sqrt(1 + a^2 / 2) on a
    k x a x orientation grid, phi = 2 pi j / (n_orient k) spanning one lobe period.
    n_shapes and rng are ignored."""
    k, a, j = (np.array(v) for v in zip(*product(ks, amps, range(n_orient))))
    phi = 2 * np.pi * j / (n_orient * k)
    kc, ac, pc = k[:, None], a[:, None], phi[:, None]
    radii = r0 * (1 + ac * np.cos(kc * (THETA - pc))) / np.sqrt(1 + ac**2 / 2)
    _, margins = is_convex(radii)
    masks = np.array([shape_mask(r, Ny, Nx, cx, cy) for r in radii])
    return masks, radii, margins, {"k": k, "amp": a, "phi": phi}


def _polygon_dist(px, py, corners):
    """Distance from points to a convex CCW polygon, 0 inside."""
    inside = np.ones(px.shape, dtype=bool)
    dist = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in zip(corners, np.roll(corners, -1, axis=0)):
        ex, ey = bx - ax, by - ay
        qx, qy = px - ax, py - ay
        inside &= (ex * qy - ey * qx) >= 0
        t = np.clip((qx * ex + qy * ey) / (ex**2 + ey**2), 0, 1)
        dist = np.minimum(dist, np.hypot(qx - t * ex, qy - t * ey))
    return np.where(inside, 0.0, dist)


def polygon_corners(n, rho, phi, r0):
    """(n, 2) vertices of the inner n-gon that, grown by rho * r0, has area pi r0^2.
    They are the rounding centres, in the same units as r0, relative to the body centre."""
    rho_c = rho * r0
    # area of the grown n-gon: A(R) + P(R) rho + pi rho^2 = pi r0^2, solve for R
    qa = n / 2 * np.sin(2 * np.pi / n)
    qb = 2 * n * np.sin(np.pi / n) * rho_c
    qc = np.pi * (rho_c**2 - r0**2)
    R = (-qb + np.sqrt(qb**2 - 4 * qa * qc)) / (2 * qa)
    ang = phi + 2 * np.pi * np.arange(n) / n
    return np.column_stack([R * np.cos(ang), R * np.sin(ang)])


def sample_polygon_grid(
    n_shapes, Ny, Nx, cx, cy, r0, rng,
    sides=(3, 4, 6), rhos=(0.0, 0.025, 0.05, 0.075), n_orient=10,
):
    """Regular polygons with corners rounded to radius rho * r0, on a sides x rho x
    orientation grid, phi = 2 pi j / (n_orient n) spanning one symmetry period. Each is an
    inner n-gon grown by rho, sized so the area stays pi r0^2. n_shapes and rng are ignored."""
    y, x = np.mgrid[0:Ny, 0:Nx]
    px, py = x - cx, y - cy
    masks, radii, params = [], [], []
    for n, rho, j in product(sides, rhos, range(n_orient)):
        phi = 2 * np.pi * j / (n_orient * n)
        rho_c = rho * r0  # rounding radius in cells
        corners = polygon_corners(n, rho, phi, r0)
        R = np.hypot(*corners[0])

        masks.append(_polygon_dist(px, py, corners) <= rho_c)

        # boundary radius along each THETA by bisection (the shape is convex around 0)
        lo, hi = np.zeros(N_THETA), np.full(N_THETA, R + rho_c)
        for _ in range(40):
            mid = (lo + hi) / 2
            out = _polygon_dist(mid * np.cos(THETA), mid * np.sin(THETA), corners) > rho_c
            hi, lo = np.where(out, mid, hi), np.where(out, lo, mid)
        radii.append((lo + hi) / 2)
        params.append((n, rho, phi))

    n, rho, phi = (np.array(v) for v in zip(*params))
    return (
        np.array(masks),
        np.array(radii),
        np.zeros(len(masks)),  # all convex
        {"sides": n, "rho": rho, "phi": phi},
    )
