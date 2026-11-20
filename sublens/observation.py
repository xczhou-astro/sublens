"""Instrument-independent lensing observations and source reconstruction results."""

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
import torch
from astropy.io import fits

from .inversion.source_plane import PixelSource, PixelSourceProblem, fit_pixel_source
from .grids import make_image_rays


def _numpy(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return array if array.dtype.isnative else array.astype(array.dtype.newbyteorder("="))


def _macro_defaults(model):
    parameters = dict(theta_E=1.4, q=0.8, phi=0.0, gamma=2.0,
                      gamma_ext=0.0, phi_ext=0.0, center=(0.0, 0.0))
    parameters.update(model)
    parameters["center"] = tuple(parameters["center"])
    return parameters


@dataclass
class LensingObservation:
    lens_id: str
    image: np.ndarray
    rms: np.ndarray
    psf: np.ndarray
    mask: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    pixel_scale: float
    lens_center: Tuple[float, float] = (0.0, 0.0)
    einstein_radius: Optional[float] = None
    native_rays: Optional[Tuple[np.ndarray, np.ndarray]] = None
    macro_model: Optional[dict] = None
    lens_light: Optional[np.ndarray] = None
    raw_image: Optional[np.ndarray] = None

    @property
    def noise(self):
        """Per-pixel noise standard deviation in the same units as image."""
        return self.rms

    @classmethod
    def from_arrays(cls, image, noise, psf, *, pixel_scale, macro_model=None,
                    alpha_macro=None, mask=None, bx=None, by=None, oversample=1,
                    native_rays=None, lens_id="observation", lens_center=(0.0, 0.0),
                    einstein_radius=None, lens_light=None, image_is_lens_subtracted=False):
        """Create an observation from arrays and a macro model, deflection or source rays.

        noise is a standard-deviation map in image units. PSF uses convolution.
        Angular coordinates are in arcsec, centered on the image, with x along
        increasing columns and y along increasing rows unless native_rays is supplied.
        """
        image = _numpy(image)
        if image.ndim == 4 and image.shape[:2] == (1, 1):
            image = image[0, 0]
        if image.ndim != 2:
            raise ValueError("Image must be a two-dimensional array")
        noise = _numpy(noise)
        if noise.ndim == 4 and noise.shape[:2] == (1, 1):
            noise = noise[0, 0]
        rms = np.broadcast_to(noise, image.shape).copy()
        psf = _numpy(psf)
        if psf.ndim == 4 and psf.shape[:2] == (1, 1):
            psf = psf[0, 0]
        mask = np.ones(image.shape, dtype=bool) if mask is None else np.asarray(_numpy(mask), dtype=bool)
        foreground=raw=None
        if lens_light is not None:
            foreground=_numpy(lens_light)
            if foreground.ndim==4 and foreground.shape[:2]==(1,1):foreground=foreground[0,0]
            if foreground.shape!=image.shape or mask.shape!=image.shape or not np.isfinite(foreground[mask]).all():
                raise ValueError('lens_light must match the image and be finite on valid pixels')
            foreground=np.array(foreground,copy=True)
            raw=np.array(image+foreground if image_is_lens_subtracted else image,copy=True)
            if not image_is_lens_subtracted:image=np.asarray(image,dtype=float)-foreground
        if (mask.shape != image.shape or not mask.any() or not np.isfinite(image[mask]).all()
                or not np.isfinite(rms[mask]).all() or np.any(rms[mask] <= 0)):
            raise ValueError("Valid pixels need finite image values and positive finite noise RMS")
        if (psf.ndim != 2 or any(n % 2 == 0 for n in psf.shape)
                or not np.isfinite(psf).all() or psf.sum() <= 0):
            raise ValueError("PSF must have odd dimensions, finite values and a positive sum")
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive in arcsec")
        rays = make_image_rays(image.shape, pixel_scale, oversample) if native_rays is None else native_rays
        x, y = (np.asarray(_numpy(a)) for a in rays)
        if x.ndim != 3 or x.shape[:2] != image.shape or y.shape != x.shape:
            raise ValueError("Image rays must have shape (rows, columns, samples)")
        model = None if macro_model is None else _macro_defaults(macro_model)
        if model is not None:
            lens_center, einstein_radius = model["center"], model["theta_E"]
        if (bx is None) != (by is None):
            raise ValueError("Supply both source ray arrays bx and by")
        if bx is None:
            if alpha_macro is not None:
                alpha = _numpy(alpha_macro)
                if alpha.size != x.size * 2:
                    raise ValueError("Macro deflection must have two components per image ray")
                alpha = alpha.reshape(*x.shape, 2)
            elif model is not None:
                from .profiles.macro import epl_deflection, external_shear_deflection

                coords = torch.as_tensor(np.column_stack((x.ravel(), y.ravel())), dtype=torch.float64)
                host, _ = epl_deflection(
                    coords, theta_E=model["theta_E"], q=model["q"], phi=model["phi"],
                    gamma=model["gamma"], center=model["center"],
                )
                shear = external_shear_deflection(
                    coords, gamma_ext=model["gamma_ext"], phi_ext=model["phi_ext"], center=model["center"],
                )
                alpha = (host + shear).numpy().reshape(*x.shape, 2)
            else:
                raise ValueError("Supply macro_model, alpha_macro, or source rays bx and by")
            bx, by = x - alpha[..., 0], y - alpha[..., 1]
        bx, by = np.asarray(_numpy(bx)), np.asarray(_numpy(by))
        if bx.shape != x.shape or by.shape != x.shape or not np.isfinite([bx, by, x, y]).all():
            raise ValueError("Image and source rays must have matching finite coordinates")
        return cls(lens_id, image, rms, psf / psf.sum(), mask, bx, by, float(pixel_scale),
                   tuple(lens_center), einstein_radius, (x, y), model,foreground,raw)

    @classmethod
    def from_fits(cls, path, *, noise=None, psf=None, pixel_scale=None, macro_model=None, mask=None,lens_light=None):
        """Read a bundled FITS observation or supply separate RMS and PSF arrays/files."""
        from .io import MACRO_CARDS, read_lensing_fits, pixel_scale_from_header, foreground_input

        data = read_lensing_fits(path, noise=noise, psf=psf)
        header = data["header"]
        pixel_scale=pixel_scale_from_header(header,pixel_scale)
        image,foreground,subtracted=foreground_input(data,lens_light)
        override_macro = macro_model is not None
        if macro_model is None and "MACRO_TE" in header:
            macro_model = {key: header[card] for key, card in MACRO_CARDS.items() if card in header}
            macro_model["center"] = (header.get("MACRO_XC", 0.0), header.get("MACRO_YC", 0.0))
        if ("beta_x" in data) != ("beta_y" in data) or ("theta_x" in data) != ("theta_y" in data):
            raise ValueError("FITS ray coordinates must contain both X and Y components")
        native = (data["theta_x"], data["theta_y"]) if "theta_x" in data else None
        bx, by = data.get("beta_x"), data.get("beta_y")
        if override_macro:
            bx = by = None
        return cls.from_arrays(
            image, data["noise"], data["psf"], pixel_scale=pixel_scale,
            macro_model=macro_model, mask=data.get("mask") if mask is None else mask,
            bx=bx, by=by, native_rays=native, oversample=int(header.get("OVERSAMP", 1)),
            lens_id=header.get("OBJECT", "observation"),
            lens_center=(header.get("LENS_X", 0.0), header.get("LENS_Y", 0.0)),
            einstein_radius=header.get("THETA_E"),
            lens_light=foreground,image_is_lens_subtracted=subtracted,
        )

    def with_macro(self, macro_model):
        """Remap the same observation, retaining raw pixels and foreground provenance."""
        updated=type(self).from_arrays(self.image,self.rms,self.psf,pixel_scale=self.pixel_scale,
            macro_model=macro_model,mask=self.mask,native_rays=self.image_rays(),lens_id=self.lens_id)
        return replace(updated,lens_light=self.lens_light,raw_image=self.raw_image)

    def to_fits(self, path, *, overwrite=False):
        """Save a reusable SCI/RMS/PSF/MASK FITS bundle with exact ray geometry."""
        from .io import write_lensing_fits

        write_lensing_fits(self, path, overwrite=overwrite)

    @classmethod
    def from_simulation(cls, simulator, image, alpha_macro, *, mask=None,
                        lens_id="simulation", lens_center=(0.0, 0.0), einstein_radius=None,
                        macro_model=None):
        """Wrap a simulator image and its fixed macro deflection on native pixel centers."""
        size = simulator.N_img
        image = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        if image.shape == (1, 1, size, size):
            image = image[0, 0]
        if image.shape != (size, size):
            raise ValueError("Image shape must match the simulator grid")
        alpha = torch.as_tensor(alpha_macro, device=simulator.device,
                                dtype=simulator.theta_grid.dtype)
        if alpha.shape != simulator.theta_grid.shape:
            raise ValueError("Macro deflection must have shape (number of pixels, 2)")
        beta = (simulator.theta_grid - alpha).detach().cpu().numpy().reshape(size, size, 1, 2)
        theta = simulator.theta_grid.detach().cpu().numpy().reshape(size, size, 1, 2)
        rms = simulator.noise_map[0, 0].detach().cpu().numpy()
        # Simulator conv2d uses cross-correlation. Observation PSFs use convolution.
        psf = simulator.psf_kernel[0, 0].detach().cpu().numpy()[::-1, ::-1].copy()
        mask = np.ones(image.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError("Mask shape must match the observation")
        return cls.from_arrays(
            image, rms, psf, mask=mask, bx=beta[..., 0], by=beta[..., 1],
            pixel_scale=simulator.pixel_scale, native_rays=(theta[..., 0], theta[..., 1]),
            lens_id=lens_id, lens_center=lens_center, einstein_radius=einstein_radius,
            macro_model=macro_model,
        )

    def image_rays(self):
        """Return the native sky coordinate of each subpixel ray."""
        if self.native_rays is not None:
            return self.native_rays
        oversample = math.isqrt(self.bx.shape[-1])
        if oversample**2 != self.bx.shape[-1]:
            raise ValueError("An observation needs a square subpixel grid")
        return make_image_rays(self.image.shape, self.pixel_scale, oversample)

    def reconstruct(self, *, fit_mask=None, backend=None, device=None, **source_options):
        """Fit the source and predict all image pixels, including held-out pixels."""
        from .execution import selected_execution
        with selected_execution(backend, device):
            mask = self._fit_mask(fit_mask)
            source = fit_pixel_source(
                self.image, self.rms, self.psf, mask, self.bx, self.by, **source_options
            )
            return self._source_result(source, mask)

    def reconstruct_delaunay(self, *, fit_mask=None, validation_mask=None,
                             seeds=(11, 22, 33, 44), device="cpu",
                             max_steps=2000, tolerance=1e-5, backend=None, **mesh_options):
        """Fit one PyTorch source per mesh seed and select by held-out chi-square.

        If only validation_mask is given, fit the other valid pixels.
        Without validation pixels, selection uses the fitted pixels.
        """
        if backend is not None:
            if backend != "torch":
                raise ValueError("Explicit Delaunay backend must be 'torch'")
            from .execution import torch_backend
            with torch_backend(device, source_tolerance=tolerance, source_max_steps=max_steps):
                return self.reconstruct_delaunay(fit_mask=fit_mask, validation_mask=validation_mask,
                    seeds=seeds, device=device, max_steps=max_steps, tolerance=tolerance, **mesh_options)
        mask = self._fit_mask(fit_mask)
        seeds = tuple(seeds)
        if not seeds:
            raise ValueError("Supply at least one Delaunay mesh seed")
        validation = None
        if validation_mask is not None:
            validation = np.asarray(validation_mask, dtype=bool)
            if (validation.shape != self.mask.shape or not validation.any()
                    or np.any(validation & ~self.mask)):
                raise ValueError("Validation pixels must be a nonempty subset of valid pixels")
            if fit_mask is None:
                mask = mask & ~validation
            elif np.any(validation & mask):
                raise ValueError("Validation and fit masks must be disjoint")
            if (not np.isfinite(self.image[validation]).all()
                    or not np.isfinite(self.rms[validation]).all()
                    or np.any(self.rms[validation] <= 0)):
                raise ValueError("Validation pixels need finite flux and positive finite RMS")

        runs = []
        for seed in seeds:
            problem = PixelSourceProblem(
                self.image, self.rms, self.psf, mask, self.bx, self.by,
                method="delaunay", seed=seed, **mesh_options,
            )
            source = problem.fit_torch(
                self.bx, self.by, device=device, max_steps=max_steps, tolerance=tolerance,
            )
            result = self._source_result(source, mask)
            train_chi2 = float(np.mean((result.residual[mask] / self.rms[mask])**2))
            validation_chi2 = None
            if validation is not None:
                validation_chi2 = float(np.mean((result.residual[validation] / self.rms[validation])**2))
            runs.append(DelaunayRun(seed, result, train_chi2, validation_chi2))

        def score(run):
            return run.train_chi2_per_pixel if validation is None else run.validation_chi2_per_pixel

        return DelaunayReconstruction(tuple(runs), min(runs, key=score))

    def search_pinn(self, *, backend=None, device=None, **options):
        """Search for compact halos with the PINN proposal and continuous refinement."""
        from .inversion.halo_search import search_pinn_subhalos

        from .execution import selected_execution
        with selected_execution(backend, device):
            return search_pinn_subhalos(self, **options)

    def fit_macro_torch(self, *, device=None, **options):
        """Profile a positive Delaunay source while fitting seven macro parameters."""
        from .execution import current_execution, torch_backend
        from .inversion.torch_refinement import fit_macro_torch
        if device is None and current_execution() is not None:
            return fit_macro_torch(self, **options)
        with torch_backend(device or "cpu"):
            return fit_macro_torch(self, **options)

    def _fit_mask(self, fit_mask):
        mask = self.mask if fit_mask is None else np.asarray(fit_mask, dtype=bool)
        if mask.shape != self.mask.shape or np.any(mask & ~self.mask):
            raise ValueError("A fit mask must be a subset of the valid observation aperture")
        return mask

    def _source_result(self, source, mask):
        return SourceReconstructionResult(
            self.lens_id, source, mask, self.image - source.image_model, self.rms, self.pixel_scale,
            math.isqrt(self.bx.shape[-1]),
        )


@dataclass
class DelaunayRun:
    seed: int
    result: "SourceReconstructionResult"
    train_chi2_per_pixel: float
    validation_chi2_per_pixel: Optional[float]


@dataclass
class DelaunayReconstruction:
    runs: Tuple[DelaunayRun, ...]
    best: DelaunayRun

    @property
    def source(self):
        return self.best.result.source

    def plot(self, **options):
        """Plot the best reconstruction and return the Matplotlib figure."""
        return self.best.result.plot(**options)


@dataclass
class SourceReconstructionResult:
    lens_id: str
    source: PixelSource
    fit_mask: np.ndarray
    residual: np.ndarray
    rms: np.ndarray
    pixel_scale: float
    oversample: int

    def plot(self, *, source_half_width=0.65, resolution=201, show_mesh=True, save_path=None):
        """Plot the source, observed image, PSF-convolved fit and normalized residual."""
        from .utils.visualization import plot_source_reconstruction

        return plot_source_reconstruction(
            self, source_half_width=source_half_width, resolution=resolution,
            show_mesh=show_mesh, save_path=save_path,
        )

    def source_image(self, half_width=1.0, size=201):
        """Rasterize the source, with increasing y in increasing FITS rows."""
        if not np.isfinite(half_width) or half_width <= 0 or size < 2:
            raise ValueError("Source raster needs positive half-width and at least two pixels")
        axis = np.linspace(-half_width, half_width, size)
        return self.source.sample(*np.meshgrid(axis, axis))

    def save_fits(self, path, *, half_width=1.0, size=201, overwrite=False):
        pixels = self.source_image(half_width, size)
        header = fits.Header()
        header["OBJECT"] = self.lens_id
        header["METHOD"] = self.source.method
        header["SRCREG"] = (self.source.regularization, "Source regularization coefficient")
        header["NCOEFF"] = len(self.source.coefficients)
        header["OVERSAMP"] = (self.oversample, "Ray samples per axis in each native pixel")
        header["IMGSCALE"] = (self.pixel_scale, "Native image pixel scale in arcsec")
        for i in (1, 2):
            header[f"CTYPE{i}"] = "LINEAR"
            header[f"CUNIT{i}"] = "arcsec"
            header[f"CRPIX{i}"] = (size + 1) / 2
            header[f"CRVAL{i}"] = 0.0
            header[f"CDELT{i}"] = 2 * half_width / (size - 1)
        source = fits.PrimaryHDU(pixels, header=header)
        source.header["EXTNAME"] = "SOURCE"
        hdus = [source, fits.ImageHDU(self.source.image_model, name="MODEL"),
                fits.ImageHDU(self.residual, name="RESIDUAL"),
                fits.ImageHDU(self.rms, name="RMS"),
                fits.ImageHDU(self.fit_mask.astype(np.uint8), name="MASK")]
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)
