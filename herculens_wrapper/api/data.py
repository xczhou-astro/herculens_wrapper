"""Single-band input data containers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class SingleBandData:
    image: np.ndarray
    noise: np.ndarray
    psf: np.ndarray
    pixel_scale: float
    psf_supersampling_factor: int = 1
    crop_size: int | None = None
    background_subtract: dict[str, Any] = field(
        default_factory=lambda: {"num_pixels": 0, "corner": "upper left"}
    )
    source_arc_mask_path: str | None = None
    source_arc_mask_radius: dict | None = None
    contaminate_mask_path: str | None = None
    background_offset: float = field(init=False, default=0.0)
    _input_paths: dict[str, Path] = field(init=False, repr=False, default_factory=dict)

    def __setattr__(self, name, value) -> None:
        """Keep cached masks consistent when notebook users edit data settings."""
        if name == "pixel_scale":
            if float(value) <= 0:
                raise ValueError("pixel_scale must be positive.")
            object.__setattr__(self, name, float(value))
            if "_source_arc_mask" in self.__dict__ and self.source_arc_mask_path is None:
                object.__setattr__(self, "_source_arc_mask", None)
            return
        if name == "psf_supersampling_factor":
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError("psf_supersampling_factor must be a positive integer.")
            object.__setattr__(self, name, int(value))
            return
        if name == "source_arc_mask_radius":
            if value is not None and not isinstance(value, dict):
                raise TypeError("source_arc_mask_radius must be a dict, for example {'inner': 0.2, 'outer': 0.8}.")
            object.__setattr__(self, name, value)
            if "_source_arc_mask" in self.__dict__:
                object.__setattr__(self, "_source_arc_mask", None)
            return
        if name == "source_arc_mask_path":
            object.__setattr__(self, name, value)
            if "_source_arc_mask" in self.__dict__:
                object.__setattr__(self, "_source_arc_mask", None)
            return
        if name == "contaminate_mask_path":
            object.__setattr__(self, name, value)
            if "_contaminate_mask" in self.__dict__:
                object.__setattr__(self, "_contaminate_mask", None)
            return
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.image, self.noise, self.psf = np.asarray(self.image, dtype=float), np.asarray(self.noise, dtype=float), np.asarray(self.psf, dtype=float)
        if self.image.ndim != 2 or self.image.shape[0] != self.image.shape[1]: raise ValueError("image must be a square 2-D array.")
        if self.noise.shape != self.image.shape: raise ValueError("noise must have the same shape as image.")
        if self.psf.ndim != 2: raise ValueError("psf must be a 2-D array.")
        self._apply_crop()
        self._apply_background_subtraction()
        self._source_arc_mask: np.ndarray | None = None
        self._contaminate_mask: np.ndarray | None = None

    def _apply_crop(self) -> None:
        if self.crop_size is None:
            return
        if not isinstance(self.crop_size, int) or isinstance(self.crop_size, bool) or self.crop_size <= 0:
            raise ValueError("crop_size must be a positive integer or None.")
        if self.crop_size > min(self.image.shape):
            raise ValueError(f"crop_size={self.crop_size} exceeds image shape {self.image.shape}.")
        from herculens_wrapper.utils import center_crop
        self.image, self.noise = center_crop(self.image, self.crop_size), center_crop(self.noise, self.crop_size)

    def _apply_background_subtraction(self) -> None:
        settings = self.background_subtract
        if not isinstance(settings, dict):
            raise TypeError("background_subtract must be a dictionary.")
        num_pixels = settings.get("num_pixels", 0)
        if not isinstance(num_pixels, int) or isinstance(num_pixels, bool) or num_pixels < 0:
            raise ValueError("background_subtract['num_pixels'] must be a non-negative integer.")
        corner = str(settings.get("corner", "upper left")).lower().replace("_", " ").replace("-", " ").strip()
        aliases = {"upper left": "upper left", "top left": "upper left", "upper right": "upper right", "top right": "upper right", "lower left": "lower left", "bottom left": "lower left", "lower right": "lower right", "bottom right": "lower right"}
        if corner not in aliases:
            raise ValueError("background_subtract['corner'] must name an upper/lower left/right corner.")
        self.background_subtract = {"num_pixels": num_pixels, "corner": aliases[corner]}
        if num_pixels == 0:
            self.background_offset = 0.0
            return
        if num_pixels > min(self.image.shape):
            raise ValueError(f"background_subtract['num_pixels']={num_pixels} exceeds image shape {self.image.shape}.")
        c = num_pixels
        regions = {
            "upper left": self.image[-c:, :c], "upper right": self.image[-c:, -c:],
            "lower left": self.image[:c, :c], "lower right": self.image[:c, -c:],
        }
        self.background_offset = float(np.nanmedian(regions[aliases[corner]]))
        self.image = self.image - self.background_offset

    @classmethod
    def from_fits(cls, image_path: str | Path, noise_path: str | Path, psf_path: str | Path, *, pixel_scale: float,
                  psf_supersampling_factor: int = 1,
                  crop_size: int | None = None, background_subtract: dict[str, Any] | None = None,
                  source_arc_mask_path: str | None = None, source_arc_mask_radius: dict | None = None,
                  contaminate_mask_path: str | None = None) -> "SingleBandData":
        from astropy.io import fits
        instance = cls(
            image=fits.getdata(image_path), noise=fits.getdata(noise_path), psf=fits.getdata(psf_path),
            pixel_scale=pixel_scale, psf_supersampling_factor=psf_supersampling_factor,
            crop_size=crop_size, background_subtract=(
                {"num_pixels": 0, "corner": "upper left"} if background_subtract is None else background_subtract
            ), source_arc_mask_path=source_arc_mask_path, source_arc_mask_radius=source_arc_mask_radius,
            contaminate_mask_path=contaminate_mask_path,
        )
        instance._input_paths = {
            "image": Path(image_path).expanduser(),
            "noise": Path(noise_path).expanduser(),
            "psf": Path(psf_path).expanduser(),
        }
        if source_arc_mask_path is not None:
            instance._input_paths["source_arc_mask"] = Path(source_arc_mask_path).expanduser()
        if contaminate_mask_path is not None:
            instance._input_paths["contaminate_mask"] = Path(contaminate_mask_path).expanduser()
        return instance

    @classmethod
    def from_fits_hdus(
        cls,
        fits_path: str | Path,
        *,
        pixel_scale: float,
        image_hdu: str | int = 0,
        noise_hdu: str | int = "NOISE",
        psf_hdu: str | int = "PSF",
        source_arc_mask_hdu: str | int | None = None,
        contaminate_mask_hdu: str | int | None = None,
        psf_supersampling_factor: int = 1,
        crop_size: int | None = None,
        background_subtract: dict[str, Any] | None = None,
        source_arc_mask_radius: dict | None = None,
    ) -> "SingleBandData":
        """Load image products stored as HDUs in one FITS container.

        Each ``*_hdu`` may be an integer HDU index or an extension name.  By
        default the science image is read from the primary HDU, and ``NOISE``
        and ``PSF`` extensions are used.  Masks are optional; an in-memory
        mask from an HDU takes precedence over ``source_arc_mask_radius``.

        Unlike :meth:`from_fits`, this constructor does not retain individual
        input filenames because all products originate from one container.
        Consequently :meth:`save` writes the usual separate ``image.fits``,
        ``noise.fits``, etc. processed products.
        """
        from astropy.io import fits

        container = Path(fits_path).expanduser()
        if not container.is_file():
            raise FileNotFoundError(f"FITS container not found: {container}")

        def read_hdu(hdul, selector: str | int, role: str) -> np.ndarray:
            try:
                data = hdul[selector].data
            except (KeyError, IndexError, TypeError) as error:
                available = ", ".join(
                    f"{index}:{hdu.name}" for index, hdu in enumerate(hdul)
                )
                raise ValueError(
                    f"Could not find {role} HDU {selector!r} in {container}. "
                    f"Available HDUs: {available}."
                ) from error
            if data is None:
                raise ValueError(f"{role} HDU {selector!r} in {container} contains no image data.")
            return np.asarray(data)

        with fits.open(container, memmap=False) as hdul:
            image = read_hdu(hdul, image_hdu, "image")
            noise = read_hdu(hdul, noise_hdu, "noise")
            psf = read_hdu(hdul, psf_hdu, "PSF")
            source_mask = (
                None if source_arc_mask_hdu is None
                else read_hdu(hdul, source_arc_mask_hdu, "source-arc mask")
            )
            contaminate_mask = (
                None if contaminate_mask_hdu is None
                else read_hdu(hdul, contaminate_mask_hdu, "contaminate mask")
            )

        instance = cls(
            image=image, noise=noise, psf=psf, pixel_scale=pixel_scale,
            psf_supersampling_factor=psf_supersampling_factor,
            crop_size=crop_size,
            background_subtract=(
                {"num_pixels": 0, "corner": "upper left"}
                if background_subtract is None else background_subtract
            ),
            source_arc_mask_radius=source_arc_mask_radius,
        )
        # A mask held in the container must receive exactly the same crop as
        # the data, but should not be re-read as a primary-HDU FITS file.
        if source_mask is not None:
            instance._source_arc_mask = instance._coerce_mask_array(
                source_mask, "source arc mask", binary=False,
            )
        if contaminate_mask is not None:
            instance._contaminate_mask = instance._coerce_mask_array(
                contaminate_mask, "contaminate mask", binary=True,
            )
        return instance

    def save(self, save_path: str | Path) -> dict[str, Path]:
        """Save processed data and any active masks as FITS files.

        Original FITS basenames are retained when the data came from
        :meth:`from_fits`; otherwise ``image.fits``, ``noise.fits``, and
        ``psf.fits`` are used. The saved image includes API preprocessing
        such as cropping and background subtraction. Source-arc and
        contamination masks are also saved when configured; masks generated
        from ``source_arc_mask_radius`` use ``source_arc_mask.fits``.
        """
        from astropy.io import fits

        directory = Path(save_path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        names = {
            key: self._input_paths.get(key, Path(f"{key}.fits")).name
            for key in ("image", "noise", "psf")
        }
        arrays = {"image": self.image, "noise": self.noise, "psf": self.psf}
        output = {key: directory / name for key, name in names.items()}
        for key, filename in output.items():
            header = None
            if key == "psf":
                header = fits.Header()
                header["PSFSSAMP"] = (
                    self.psf_supersampling_factor,
                    "PSF supersampling relative to image pixels",
                )
            fits.writeto(filename, np.asarray(arrays[key]), header=header, overwrite=True)
        if self.source_arc_mask is not None:
            source_name = self._input_paths.get(
                "source_arc_mask", Path("source_arc_mask.fits"),
            ).name
            output["source_arc_mask"] = directory / source_name
            fits.writeto(
                output["source_arc_mask"], np.asarray(self.source_arc_mask, dtype=np.uint8),
                overwrite=True,
            )
        if self.contaminate_mask is not None:
            contaminate_name = self._input_paths.get(
                "contaminate_mask", Path("contaminate_mask.fits"),
            ).name
            output["contaminate_mask"] = directory / contaminate_name
            fits.writeto(
                output["contaminate_mask"], np.asarray(self.contaminate_mask, dtype=np.uint8),
                overwrite=True,
            )
        return output

    def _load_mask(self, path: str, role: str, *, binary: bool) -> np.ndarray:
        from astropy.io import fits
        return self._coerce_mask_array(fits.getdata(path), role, binary=binary)

    def _coerce_mask_array(self, mask, role: str, *, binary: bool) -> np.ndarray:
        """Apply data preprocessing and validation to a loaded mask array."""
        mask = np.asarray(mask)
        if self.crop_size is not None:
            from herculens_wrapper.utils import center_crop
            mask = center_crop(mask, self.crop_size)
        if mask.shape != self.image.shape:
            raise ValueError(f"{role} shape {mask.shape} does not match image shape {self.image.shape}.")
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"{role} must contain finite values.")
        if binary and not np.all(np.isclose(mask, 0.0) | np.isclose(mask, 1.0)):
            raise ValueError(f"{role} must contain only 0 and 1 values.")
        return mask.astype(bool)

    @property
    def source_arc_mask(self) -> np.ndarray | None:
        """Source-arc support mask from FITS, or a centred annulus from radii."""
        if self._source_arc_mask is None:
            if self.source_arc_mask_path is not None:
                self._source_arc_mask = self._load_mask(self.source_arc_mask_path, "source arc mask", binary=False)
            elif self.source_arc_mask_radius is not None:
                from herculens_wrapper.utils import create_source_arc_mask_from_radius
                self._source_arc_mask = create_source_arc_mask_from_radius(
                    self.image.shape, self.pixel_scale, self.source_arc_mask_radius,
                )
        return self._source_arc_mask

    @property
    def contaminate_mask(self) -> np.ndarray | None:
        """Boolean mask of contaminant pixels excluded from the likelihood."""
        if self._contaminate_mask is None and self.contaminate_mask_path is not None:
            self._contaminate_mask = self._load_mask(self.contaminate_mask_path, "contaminate mask", binary=True)
        return self._contaminate_mask

    @property
    def likelihood_mask(self) -> np.ndarray | None:
        contaminate = self.contaminate_mask
        return None if contaminate is None else ~contaminate

    @property
    def likelihood_image(self) -> np.ndarray:
        mask = self.likelihood_mask
        return np.where(mask, self.image, 0.0) if mask is not None else self.image

    @property
    def likelihood_noise(self) -> np.ndarray:
        mask = self.likelihood_mask
        return np.where(mask, self.noise, 1e10) if mask is not None else self.noise

    def show(self, *, scale: str = "linear", residual_vis_max: float = 0.0,
             save_path: str | Path | None = None):
        """Display image, noise, signal-to-noise, and PSF; optionally save.

        Parameters
        ----------
        scale
            Either ``'linear'`` or ``'log'``.  In log mode, signed images use
            symmetric-log scaling so background-subtracted data remain visible.
        save_path
            Destination image path.  ``None`` (the default) does not save.
        """
        from .visualization import plot_single_band_data
        return plot_single_band_data(self, scale=scale, residual_vis_max=residual_vis_max, save_path=save_path)
