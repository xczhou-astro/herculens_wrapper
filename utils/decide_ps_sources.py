import argparse
import functools
import json
import sys
import os
import shlex
from pathlib import Path
from typing import Any
from astropy.io import fits


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

from helens import LensEquationSolver

from herculens.MassModel.mass_model import MassModel
from herculens.Coordinates.pixel_grid import PixelGrid

class Tee:
    """Class to duplicate output to both stdout and a file"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def json_serializer(obj):

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import jax

        if isinstance(obj, jax.Array):
            return obj.tolist()
    except ImportError:
        pass
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

def center_crop(image, crop_size):

    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    h, w = image.shape[:2]
    start_y = max((h - crop_h) // 2, 0)
    start_x = max((w - crop_w) // 2, 0)
    end_y = start_y + crop_h
    end_x = start_x + crop_w

    return image[start_y:end_y, start_x:end_x]


def get_fits_data(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data.astype(np.float64)

def create_pixel_grids(npix, pix_scl):

    half_size = npix * pix_scl / 2
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2  # position of the (0, 0) with respect to bottom left pixel
    transform_pix2angle = pix_scl * np.eye(2)  # transformation matrix pixel <-> angle
    kwargs_pixel = {'nx': npix, 'ny': npix,
                    'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                    'transform_pix2angle': transform_pix2angle}
    
    pixel_grid = PixelGrid(**kwargs_pixel)
    
    ps_grid_npix = 2 * npix + 1
    ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix
    ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.
    ps_grid_ra_at_xy_0 = ps_grid_dec_at_xy_0 = -ps_grid_half_size + ps_grid_pix_scl / 2.
    ps_grid_transform_pix2angle = ps_grid_pix_scl * np.eye(2)
    kwargs_ps_grid = {'nx': ps_grid_npix, 'ny': ps_grid_npix,
                    'ra_at_xy_0': ps_grid_ra_at_xy_0, 'dec_at_xy_0': ps_grid_dec_at_xy_0,
                    'transform_pix2angle': ps_grid_transform_pix2angle}
    ps_grid = PixelGrid(**kwargs_ps_grid)
    
    return pixel_grid, ps_grid

def arcsec_to_pixel(ra: Any, dec: Any, pixel_grid):
    """(ra, dec) in arcsec -> continuous pixel indices (x, y) for plotting."""
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    x_pix = []
    y_pix = []
    for r, d in zip(ra.ravel(), dec.ravel()):
        px, py = pixel_grid.map_coord2pix(r, d)
        x_pix.append(float(px))
        y_pix.append(float(py))
    return np.array(x_pix), np.array(y_pix)

def merge_nearby_points(
    x: np.ndarray, y: np.ndarray, tol_arcsec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicates within tol (arcsec), keeping first occurrence."""
    if len(x) == 0:
        return x, y
    keep = []
    used = []
    for i in range(len(x)):
        ok = True
        for j in used:
            if np.hypot(x[i] - x[j], y[i] - y[j]) < tol_arcsec:
                ok = False
                break
        if ok:
            keep.append(i)
            used.append(i)
    keep = np.array(keep, dtype=int)
    return x[keep], y[keep]


def fwhm_arcsec_to_gaussian_sigma(fwhm_arcsec: float) -> float:
    """FWHM (arcsec) → σ for an isotropic 2D Gaussian (same σ along x and y)."""
    return fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))



def render_lensed_source_gaussian(
    beta_grid_x: np.ndarray,
    beta_grid_y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    sigma_arcsec: float,
) -> np.ndarray:
    """Evaluate a circular source-plane Gaussian on ray-shot image pixels."""
    bx = np.asarray(beta_grid_x, dtype=float)
    by = np.asarray(beta_grid_y, dtype=float)
    s = sigma_arcsec
    arg = ((bx - center_x) / s) ** 2 + ((by - center_y) / s) ** 2
    img = np.exp(-0.5 * arg)
    vmax = float(np.nanmax(img))
    if np.isfinite(vmax) and vmax > 0.0:
        img = img / vmax
    return img


def circled_number_label(n: int) -> str:
    """Return a circled number (Unicode) for small IDs; plain parentheses fallback otherwise."""
    if n == 0:
        return "\u24EA"  # ⓪
    if 1 <= n <= 20:
        return chr(0x245F + n)
    if 21 <= n <= 35:
        return chr(0x3250 + (n - 20))
    if 36 <= n <= 50:
        return chr(0x32B1 + (n - 36))
    return f"({n})"


def compute_caustics_from_mass_model(mass_model, pixel_grid, kwargs_lens):
    """Compute source-plane caustics from a MassModel on the current image grid."""
    from herculens.LensImage.lens_image import LensImage
    from herculens.Instrument.psf import PSF
    from herculens.Util.model_util import critical_lines_caustics

    try:
        # Create a minimal LensImage object to interface with herculens utility
        lens_image = LensImage(pixel_grid, PSF(), lens_mass_model_class=mass_model)
        # critical_lines_caustics returns (critical_lines, caustics)
        crit_lines, caustics = critical_lines_caustics(lens_image, kwargs_lens, supersampling=5)

        # Sort caustics by the average radius of their critical lines
        # so that the tangential/inner caustic (index 0, larger critical line) comes first,
        # and the radial/outer caustic (index 1, smaller critical line) comes second.
        if len(crit_lines) == 2:
            r0 = np.mean(np.sqrt(crit_lines[0][0]**2 + crit_lines[0][1]**2))
            r1 = np.mean(np.sqrt(crit_lines[1][0]**2 + crit_lines[1][1]**2))
            if r0 < r1:
                crit_lines = [crit_lines[1], crit_lines[0]]
                caustics = [caustics[1], caustics[0]]

        return caustics
    except Exception as e:
        print(f"Could not compute caustics using herculens.Util.model_util: {e}")
        return []


def plot_catalog_image_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    image_data: np.ndarray,
    pixel_grid,
    start_row: int = 1,
    id_column: str = "id",
) -> None:
    """Image plane: data with circled catalog IDs (left panel of catalog_source_trace)."""
    if start_row < 0 or start_row >= len(df):
        raise ValueError(f"start_row must be in [0, len(df)), got {start_row} (len={len(df)})")

    sub = df.iloc[start_row:]
    has_id = id_column in df.columns
    theta_ra = sub["ra_arcsec"].to_numpy(dtype=float)
    theta_dec = sub["dec_arcsec"].to_numpy(dtype=float)

    numeric_ids: list[int] = []
    for i, (_, row) in enumerate(sub.iterrows()):
        if has_id:
            numeric_ids.append(int(row[id_column]))
        else:
            numeric_ids.append(int(sub.index[i]))
    circled_labels = [circled_number_label(k) for k in numeric_ids]

    vmin, vmax = np.percentile(image_data, [5, 99.5])
    ax.imshow(
        image_data,
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=pixel_grid.plt_extent,
    )
    for x, y, lab in zip(theta_ra, theta_dec, circled_labels):
        t = ax.text(
            x,
            y,
            lab,
            fontsize=14,
            color="cyan",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black")])
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Image plane (catalog positions)")


def visualize_catalog_traced_sources(
    df: pd.DataFrame,
    mass_model,
    kwargs_lens,
    *,
    image_data: np.ndarray | None = None,
    pixel_grid=None,
    caustics: list | None = None,
    start_row: int = 1,
    id_column: str = "id",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (14.0, 6.5),
    dpi: int = 150,
) -> tuple[plt.Figure, np.ndarray]:
    """Ray-trace catalog image-plane positions to the source plane and plot both planes with circled IDs.

    Parameters
    ----------
    df
        Catalog with columns ``ra_arcsec`` and ``dec_arcsec``. Rows before ``start_row`` are skipped
        (default ``start_row=1`` skips the first data row, index 0).
    mass_model, kwargs_lens
        Lens mass model and keyword arguments passed to ``ray_shooting``.
    image_data, pixel_grid
        If both given, the left panel shows the image with catalog positions overlaid in pixel coords.
    caustics
        Optional list of ``(beta_x_line, beta_y_line)`` curves in the source plane (e.g. from
        :func:`compute_caustics_from_mass_model`).
    start_row
        First *pandas* row index to include (default ``1`` so iteration matches "line 1" of data
        after skipping index ``0``).
    id_column
        Column used for circled labels (Unicode ①…⑳, etc.); if missing, labels use the row index.
    save_path
        If set, figure is saved to this path.

    Returns
    -------
    fig, axes
    """
    if start_row < 0 or start_row >= len(df):
        raise ValueError(f"start_row must be in [0, len(df)), got {start_row} (len={len(df)})")

    sub = df.iloc[start_row:]
    has_id = id_column in df.columns

    theta_ra = []
    theta_dec = []
    beta_x_list = []
    beta_y_list = []
    numeric_ids: list[int] = []

    theta_ra = sub["ra_arcsec"].to_numpy(dtype=float)
    theta_dec = sub["dec_arcsec"].to_numpy(dtype=float)
    bx_all, by_all = mass_model.ray_shooting(theta_ra, theta_dec, kwargs_lens)
    bx_all = np.asarray(bx_all, dtype=float).ravel()
    by_all = np.asarray(by_all, dtype=float).ravel()
    for i, (_, row) in enumerate(sub.iterrows()):
        beta_x_list.append(float(bx_all[i]))
        beta_y_list.append(float(by_all[i]))
        if has_id:
            numeric_ids.append(int(row[id_column]))
        else:
            numeric_ids.append(int(sub.index[i]))

    circled_labels = [circled_number_label(k) for k in numeric_ids]

    theta_ra = np.asarray(theta_ra, dtype=float)
    theta_dec = np.asarray(theta_dec, dtype=float)
    beta_x_arr = np.asarray(beta_x_list, dtype=float)
    beta_y_arr = np.asarray(beta_y_list, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Image plane
    ax_img = axes[0]
    if image_data is not None and pixel_grid is not None:
        plot_catalog_image_panel(
            ax_img,
            df,
            image_data=image_data,
            pixel_grid=pixel_grid,
            start_row=start_row,
            id_column=id_column,
        )
    else:
        for x, y, lab in zip(theta_ra, theta_dec, circled_labels):
            ax_img.text(
                x,
                y,
                lab,
                fontsize=14,
                color="cyan",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
            )
        ax_img.set_xlabel("ra (arcsec)")
        ax_img.set_ylabel("dec (arcsec)")
        ax_img.set_title("Image plane (arcsec)")
        ax_img.set_aspect("equal", adjustable="box")
        ax_img.grid(alpha=0.2)

    # Source plane (markers are circled-number glyphs only; orange)
    ax_src = axes[1]
    for x, y, lab in zip(beta_x_arr, beta_y_arr, circled_labels):
        ax_src.text(
            x,
            y,
            lab,
            fontsize=14,
            color="darkorange",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
    if caustics:
        for i_c, (caust_x, caust_y) in enumerate(caustics):
            if len(caustics) == 2:
                lbl = "tangential caustic (inner)" if i_c == 0 else "radial caustic (outer)"
                color = "lime" if i_c == 0 else "deepskyblue"
                ls = "-" if i_c == 0 else "--"
            else:
                lbl = "caustic" if i_c == 0 else None
                color = "lime"
                ls = "-"
            ax_src.plot(caust_x, caust_y, color=color, lw=1.2, ls=ls, alpha=0.9, label=lbl)
    ax_src.axvline(0.0, color="red", lw=0.7, linestyle="--", alpha=0.7)
    ax_src.axhline(0.0, color="red", lw=0.7, linestyle="--", alpha=0.7)
    ax_src.set_xlabel("source x (arcsec)")
    ax_src.set_ylabel("source y (arcsec)")
    ax_src.set_title("Source plane (ray-traced)")
    ax_src.set_aspect("equal", adjustable="box")
    ax_src.grid(alpha=0.2)
    if caustics:
        ax_src.legend(loc="best", fontsize=8)

    # Zoom source panel to data + caustics
    xs = list(beta_x_arr.ravel())
    ys = list(beta_y_arr.ravel())
    if caustics:
        for caust_x, caust_y in caustics:
            xs.extend(np.asarray(caust_x, dtype=float).ravel().tolist())
            ys.extend(np.asarray(caust_y, dtype=float).ravel().tolist())
    if len(xs) > 0:
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        x_min, x_max = np.nanmin(x_arr), np.nanmax(x_arr)
        y_min, y_max = np.nanmin(y_arr), np.nanmax(y_arr)
        dx = max(x_max - x_min, 0.05)
        dy = max(y_max - y_min, 0.05)
        pad = 0.15 * max(dx, dy)
        ax_src.set_xlim(x_min - pad, x_max + pad)
        ax_src.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=dpi)
    return fig, axes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lens_params_path',
        type=str,
        required=True,
        help='Path to a JSON file containing a top-level "kwargs_lens" list of per-profile '
        'parameter dictionaries (e.g. best_params.json or kwargs_result.json).',
    )
    parser.add_argument(
        '--lens_mass_profiles',
        type=str,
        nargs='+',
        default=['SIE', 'SHEAR'],
        help='Ordered list of lens mass profile names passed to MassModel '
        '(e.g. --lens_mass_profiles SIE SHEAR). Must match the order of entries in '
        '"kwargs_lens" inside --lens_params_path.',
    )
    parser.add_argument(
        '--catalog_path',
        type=str,
        required=True,
        help='Path to source catalog CSV with columns including "ra_arcsec" and "dec_arcsec".',
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to FITS image used for visualization (image plane background).',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory where the ps_test outputs (plots, JSON, logs) are written. '
        'Defaults to "<dir of --lens_params_path>/ps_test".',
    )
    parser.add_argument(
        '--pixel_scale',
        type=float,
        default=0.03,
        help='Pixel scale (arcsec/pixel) of the image used to build the PixelGrid.',
    )
    parser.add_argument(
        '--image_crop_size',
        type=int,
        default=61,
        help='Size (pixels) to center-crop the FITS image to before plotting.',
    )
    parser.add_argument(
        '--source_gaussian_fwhm_arcsec',
        type=float,
        default=0.03,
        help='FWHM (arcsec) of source-plane Gaussian.',
    )
    parser.add_argument(
        '--point_source_start_index',
        type=int,
        default=0,
        help='Zero-based row index in the original CSV catalog where point sources begin '
        '(e.g., set to 1 if the first row is the lens).',
    )

    args = parser.parse_args()

    if not os.path.exists(args.lens_params_path):
        raise FileNotFoundError(f"Lens parameters file not found: {args.lens_params_path}")
    if not os.path.exists(args.catalog_path):
        raise FileNotFoundError(f"Catalog file not found: {args.catalog_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    if args.output_dir is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(args.lens_params_path)), 'ps_test')
    else:
        save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    log_file = open(f'{save_path}/logging.txt', 'w')
    sys.stdout = Tee(sys.stdout, log_file)

    cmd = shlex.join([sys.executable, *sys.argv])
    print(f"command: {cmd}")
    print(f"cwd:     {os.getcwd()}")
    print()

    with open(args.lens_params_path, 'r') as f:
        lens_params = json.load(f)
    if 'kwargs_lens' not in lens_params:
        raise KeyError(
            f"Expected top-level 'kwargs_lens' key in {args.lens_params_path}, "
            f"found keys: {list(lens_params.keys())}"
        )
    kwargs_lens = lens_params['kwargs_lens']

    if len(kwargs_lens) != len(args.lens_mass_profiles):
        raise ValueError(
            f"--lens_mass_profiles has {len(args.lens_mass_profiles)} entries "
            f"({args.lens_mass_profiles}) but 'kwargs_lens' in {args.lens_params_path} "
            f"has {len(kwargs_lens)} components."
        )

    print(f"lens_mass_profiles: {args.lens_mass_profiles}")
    print('kwargs_lens:')
    for i, (profile, comp) in enumerate(zip(args.lens_mass_profiles, kwargs_lens)):
        print(f"Component {i} ({profile}):")
        for key, value in comp.items():
            print(f"  {key}: {value}")
        print()

    image_data = get_fits_data(args.image_path)
    image_data = center_crop(image_data, args.image_crop_size)

    mass_model = MassModel(args.lens_mass_profiles)
    ray_shoot_jit = jax.jit(lambda x, y: mass_model.ray_shooting(x, y, kwargs_lens))

    pixel_scale = args.pixel_scale
    pixel_grid, _ = create_pixel_grids(image_data.shape[0], pixel_scale)

    sigma_gaussian_arcsec = fwhm_arcsec_to_gaussian_sigma(args.source_gaussian_fwhm_arcsec)
    if sigma_gaussian_arcsec <= 0.0 or not np.isfinite(sigma_gaussian_arcsec):
        raise ValueError(
            "--source_gaussian_fwhm_arcsec must be positive and finite, "
            f"got {args.source_gaussian_fwhm_arcsec!r}"
        )

    print(
        f"[source-plane sampling] shape=gaussian profile, FWHM={args.source_gaussian_fwhm_arcsec:g} arcsec "
        f"→ σ={sigma_gaussian_arcsec:g} arcsec"
    )

    df = pd.read_csv(args.catalog_path)
    new_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
    new_row['id'] = 0
    df = pd.concat([new_row, df]).reset_index(drop=True)

    size = len(df)

    data: list[dict[str, Any]] = []
    caustics = compute_caustics_from_mass_model(mass_model, pixel_grid, kwargs_lens)
    x_grid, y_grid = pixel_grid.pixel_coordinates
    beta_grid_x, beta_grid_y = mass_model.ray_shooting(x_grid, y_grid, kwargs_lens)
    beta_grid_x = np.asarray(beta_grid_x, dtype=float)
    beta_grid_y = np.asarray(beta_grid_y, dtype=float)
    ray_shoot = functools.partial(mass_model.ray_shooting, k=None)
    solver = LensEquationSolver(x_grid, y_grid, ray_shoot)

    fig_catalog, _ = visualize_catalog_traced_sources(
        df,
        mass_model,
        kwargs_lens,
        image_data=image_data,
        pixel_grid=pixel_grid,
        caustics=caustics,
        start_row=args.point_source_start_index + 1,
        save_path=f"{save_path}/catalog_source_trace.png",
    )
    plt.close(fig_catalog)
    
    start_idx = args.point_source_start_index + 1
    for idx in range(start_idx, size):

        row = df.iloc[idx]
        theta_x = float(row['ra_arcsec'])
        theta_y = float(row['dec_arcsec'])
        
        data.append({
            'idx': idx, 
            'theta_x': theta_x,
            'theta_y': theta_y,
        })

        beta_x, beta_y = ray_shoot_jit(theta_x, theta_y)
        beta_x = float(jnp.asarray(beta_x).ravel()[0])
        beta_y = float(jnp.asarray(beta_y).ravel()[0])
        beta = jnp.array([beta_x, beta_y])
        
        data[-1]['beta_x'] = beta_x
        data[-1]['beta_y'] = beta_y

        sigma_used_arcsec = sigma_gaussian_arcsec
        # 3σ circle vertices used only for the source-plane zoom range.
        n_circle = 180
        phi = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False, dtype=float)
        r_3sigma = 3.0 * sigma_gaussian_arcsec
        beta_psf_x = beta_x + r_3sigma * np.cos(phi)
        beta_psf_y = beta_y + r_3sigma * np.sin(phi)
        data[-1]['source_sampling_shape'] = 'gaussian'
        data[-1]['source_gaussian_fwhm_arcsec'] = args.source_gaussian_fwhm_arcsec
        data[-1]['source_gaussian_sigma_arcsec'] = sigma_gaussian_arcsec
        
        gaussian_image_plane = render_lensed_source_gaussian(
            beta_grid_x,
            beta_grid_y,
            center_x=beta_x,
            center_y=beta_y,
            sigma_arcsec=sigma_used_arcsec,
        )
        data[-1]['beta_psf_x'] = beta_psf_x
        data[-1]['beta_psf_y'] = beta_psf_y

        theta_arr, beta_arr = solver.solve(
            beta,
            kwargs_lens,
            nsolutions=5,
            niter=30,
        )
        theta_xy = np.asarray(theta_arr)
        beta_check = np.asarray(beta_arr)

        pred_ra = theta_xy[:, 0]
        pred_dec = theta_xy[:, 1]
        pred_ra, pred_dec = merge_nearby_points(pred_ra, pred_dec, tol_arcsec=0.02)

        data[-1]['num_images'] = len(pred_ra)

        data[-1]['pred_ra'] = pred_ra
        data[-1]['pred_dec'] = pred_dec

        pred_psf_ra = np.empty(0, dtype=float)
        pred_psf_dec = np.empty(0, dtype=float)
        data[-1]['pred_psf_ra'] = pred_psf_ra
        data[-1]['pred_psf_dec'] = pred_psf_dec

        print(f"Starting image (arcsec): ra = {theta_x:.6f}, dec = {theta_y:.6f}")
        print(f"Traced source plane:     beta_x = {beta_x:.6f}, beta_y = {beta_y:.6f}")
        print(f"Traced-back image positions ({len(pred_ra)} after dedup, arcsec):")
        for i, (r, d) in enumerate(zip(pred_ra, pred_dec)):
            err = np.hypot(r - theta_x, d - theta_y)
            print(f"  #{i + 1}: ra = {r: .6f}, dec = {d: .6f}  (dist to start = {err:.4f} arcsec)")
        print(
            "Rendered lensed Gaussian profile on image plane "
            f"(source FWHM={args.source_gaussian_fwhm_arcsec:g} arcsec)."
        )
        print(f"Source-plane residuals at solutions (should be ~0): max |beta - beta_target| = "
            f"{np.max(np.abs(beta_check - np.array([beta_x, beta_y]))):.2e}")

        print("Catalog vs nearest traced image (arcsec):")
        for _, row in df.iterrows():
            if pd.isna(row["ra_arcsec"]):
                continue
            cr, cd = float(row["ra_arcsec"]), float(row["dec_arcsec"])
            dists = np.hypot(pred_ra - cr, pred_dec - cd)
            j = int(np.argmin(dists))
            print(
                f"  id {int(row.get('id', -1))}: min dist = {dists[j]:.4f} "
                f"(nearest traced #{j + 1})"
            )
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        plot_catalog_image_panel(
            axes[0],
            df,
            image_data=image_data,
            pixel_grid=pixel_grid,
            start_row=args.point_source_start_index + 1,
        )
        ax = axes[1]
        vmin, vmax = np.percentile(image_data, [5, 99.5])
        ax.imshow(
            image_data,
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=pixel_grid.plt_extent,
        )

        ax.scatter(theta_x, theta_y, s=120, facecolors="none", edgecolors="cyan", linewidths=2, label="start image")

        # For a peak-normalized 2D Gaussian I(r)/I(0) = exp(-r^2 / (2 sigma^2)),
        # the iso-level at n*sigma is exp(-n^2 / 2). Matplotlib sorts the levels
        # ascending, so the per-level style arrays follow the order 3σ, 2σ, 1σ.
        sigma_n = np.array([3.0, 2.0, 1.0])
        sigma_levels = np.exp(-0.5 * sigma_n ** 2)
        sigma_linestyles = ["dotted", "dashed", "solid"]
        sigma_linewidths = [0.8, 1.0, 1.4]
        sigma_color = "deepskyblue"
        ax.contour(
            gaussian_image_plane,
            levels=sigma_levels,
            colors=[sigma_color] * len(sigma_levels),
            linewidths=sigma_linewidths,
            linestyles=sigma_linestyles,
            alpha=0.9,
            extent=pixel_grid.plt_extent,
        )
        sigma_proxies: list[Line2D] = [
            Line2D(
                [0], [0],
                color=sigma_color, lw=lw, linestyle=ls,
                label=f"lensed src {int(n)}\u03c3",
            )
            for n, lw, ls in zip(
                sigma_n[::-1], sigma_linewidths[::-1], sigma_linestyles[::-1]
            )
        ]

        ax.set_xlabel("x (arcsec)")
        ax.set_ylabel("y (arcsec)")
        ax.set_title(
            f"Lens trace: source (β)=({beta_x:.3f},{beta_y:.3f})″ → {len(pred_ra)} image(s)"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(sigma_proxies)
        labels.extend(p.get_label() for p in sigma_proxies)
        ax.legend(handles, labels, loc="upper right", fontsize=9)

        # Source-plane panel: traced source position and caustics.
        ax_src = axes[2]
        ax_src.scatter(
            [beta_x],
            [beta_y],
            s=50,
            marker='*',
            label="traced source",
        )
        # Draw 1σ, 2σ, 3σ circles in the source plane.
        src_sigma_n = [1.0, 2.0, 3.0]
        src_sigma_linestyles = ["solid", "dashed", "dotted"]
        src_sigma_linewidths = [1.4, 1.0, 0.8]
        src_sigma_color = "darkorange"
        for n, ls, lw in zip(src_sigma_n, src_sigma_linestyles, src_sigma_linewidths):
            ax_src.add_patch(
                Circle(
                    (beta_x, beta_y),
                    radius=n * sigma_used_arcsec,
                    fill=False,
                    edgecolor=src_sigma_color,
                    linewidth=lw,
                    linestyle=ls,
                    alpha=0.9,
                    label=f"src {int(n)}\u03c3",
                )
            )
        for i_c, (caust_x, caust_y) in enumerate(caustics):
            if len(caustics) == 2:
                label = "tangential caustic (inner)" if i_c == 0 else "radial caustic (outer)"
                color = "lime" if i_c == 0 else "deepskyblue"
                ls = "-" if i_c == 0 else "--"
            else:
                label = "caustic" if i_c == 0 else None
                color = "lime"
                ls = "-"
            ax_src.plot(caust_x, caust_y, color=color, lw=1.2, ls=ls, label=label)
        ax_src.axvline(0.0, color="red", lw=0.7, linestyle="--")
        ax_src.axhline(0.0, color="red", lw=0.7, linestyle="--")
        ax_src.set_xlabel("source x (arcsec)")
        ax_src.set_ylabel("source y (arcsec)")
        ax_src.set_title("Source plane")
        ax_src.set_aspect("equal", adjustable="box")
        ax_src.grid(alpha=0.2)

        # Keep the source panel focused near traced β, sampled cloud, and caustics.
        xs = [beta_x]
        ys = [beta_y]
        xs.extend(np.asarray(beta_psf_x, dtype=float).ravel().tolist())
        ys.extend(np.asarray(beta_psf_y, dtype=float).ravel().tolist())
        for caust_x, caust_y in caustics:
            xs.extend(np.asarray(caust_x, dtype=float).ravel().tolist())
            ys.extend(np.asarray(caust_y, dtype=float).ravel().tolist())
        if len(xs) > 1:
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            x_min, x_max = np.nanmin(x_arr), np.nanmax(x_arr)
            y_min, y_max = np.nanmin(y_arr), np.nanmax(y_arr)
            dx = max(x_max - x_min, 0.05)
            dy = max(y_max - y_min, 0.05)
            pad = 0.15 * max(dx, dy)
            ax_src.set_xlim(x_min - pad, x_max + pad)
            ax_src.set_ylim(y_min - pad, y_max + pad)
        else:
            pad = 0.3
            ax_src.set_xlim(beta_x - pad, beta_x + pad)
            ax_src.set_ylim(beta_y - pad, beta_y + pad)
        ax_src.legend(loc="best", fontsize=9)

        fig.tight_layout()
        fig.savefig(f'{save_path}/result_ps_{idx}.png', dpi=150)
        plt.close(fig)

    with open(f'{save_path}/ps_data.json', 'w') as f:
        json.dump(data, f, indent=4, default=json_serializer)