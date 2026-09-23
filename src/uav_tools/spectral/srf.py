from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import numpy as np


ArrayLike1D = Iterable[float] | np.ndarray


def _as_1d_float_array(values: ArrayLike1D, name: str) -> np.ndarray:
    """Convert input to a 1D float numpy array."""
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like object.")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    return arr


def _validate_same_length(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    """Ensure two arrays have matching lengths."""
    if a.shape != b.shape:
        raise ValueError(
            f"{a_name} and {b_name} must have the same shape. "
            f"Got {a.shape} and {b.shape}."
        )


def _validate_strictly_increasing(arr: np.ndarray, name: str) -> None:
    """Ensure array values are strictly increasing."""
    diffs = np.diff(arr)
    if np.any(diffs <= 0):
        raise ValueError(f"{name} must be strictly increasing.")


def _validate_nonnegative(arr: np.ndarray, name: str) -> None:
    """Ensure array contains no negative values."""
    if np.any(arr < 0):
        raise ValueError(f"{name} cannot contain negative values.")


def gaussian_response(wavelengths: ArrayLike1D, center: float, fwhm: float) -> np.ndarray:
    """
    Create a Gaussian spectral response curve.

    Parameters
    ----------
    wavelengths : array-like
        Wavelength grid.
    center : float
        Center wavelength of the band.
    fwhm : float
        Full width at half maximum of the band.

    Returns
    -------
    np.ndarray
        Unnormalized Gaussian response evaluated on the wavelength grid.
    """
    wl = _as_1d_float_array(wavelengths, "wavelengths")

    if fwhm <= 0:
        raise ValueError("fwhm must be > 0.")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    response = np.exp(-((wl - center) ** 2) / (2.0 * sigma**2))
    return response


@dataclass(slots=True)
class BandSRF:
    """
    Spectral response function for a single band.

    This is the canonical internal representation used by the library:
    a wavelength array and a matching response array.

    Attributes
    ----------
    name : str
        Band name, e.g. 'red', 'nir', 'B8A'.
    wavelengths : np.ndarray
        1D wavelength grid, typically in nm.
    responses : np.ndarray
        1D relative response values on the same grid as `wavelengths`.
    definition_type : str
        How this band was originally defined, e.g.:
        'explicit_curve' or 'center_fwhm'.
    center : float | None
        Optional center wavelength metadata.
    fwhm : float | None
        Optional FWHM metadata.
    wavelength_unit : str
        Unit of wavelength values. Default is 'nm'.
    normalized : bool
        Whether the response has been area-normalized.
    provenance : str | None
        Optional source description, e.g. 'ESA CSV' or 'MicaSense band table'.
    metadata : dict[str, Any]
        Free-form metadata.
    """

    name: str
    wavelengths: np.ndarray
    responses: np.ndarray
    definition_type: str
    center: float | None = None
    fwhm: float | None = None
    wavelength_unit: str = "nm"
    normalized: bool = False
    provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelengths = _as_1d_float_array(self.wavelengths, "wavelengths")
        self.responses = _as_1d_float_array(self.responses, "responses")

        _validate_same_length(self.wavelengths, self.responses, "wavelengths", "responses")
        _validate_strictly_increasing(self.wavelengths, "wavelengths")
        _validate_nonnegative(self.responses, "responses")

        if np.all(self.responses == 0):
            raise ValueError("responses cannot be all zeros.")

        if self.center is not None and not np.isfinite(self.center):
            raise ValueError("center must be finite when provided.")

        if self.fwhm is not None and self.fwhm <= 0:
            raise ValueError("fwhm must be > 0 when provided.")

    @classmethod
    def from_curve(
        cls,
        name: str,
        wavelengths: ArrayLike1D,
        responses: ArrayLike1D,
        *,
        wavelength_unit: str = "nm",
        normalize: bool = False,
        provenance: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BandSRF:
        """
        Construct a BandSRF from explicit wavelength-response arrays.
        """
        wl = _as_1d_float_array(wavelengths, "wavelengths")
        rsp = _as_1d_float_array(responses, "responses")

        if normalize:
            area = np.trapezoid(rsp, wl)
            if area <= 0:
                raise ValueError("Cannot normalize responses with non-positive area.")
            rsp = rsp / area

        return cls(
            name=name,
            wavelengths=wl,
            responses=rsp,
            definition_type="explicit_curve",
            wavelength_unit=wavelength_unit,
            normalized=normalize,
            provenance=provenance,
            metadata=metadata or {},
        )

    @classmethod
    def from_center_fwhm(
        cls,
        name: str,
        center: float,
        fwhm: float,
        wavelengths: ArrayLike1D,
        *,
        wavelength_unit: str = "nm",
        normalize: bool = False,
        provenance: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BandSRF:
        """
        Construct a BandSRF from center wavelength + FWHM using a Gaussian approximation.

        Responses are not area-normalized unless normalize=True.
        """
        wl = _as_1d_float_array(wavelengths, "wavelengths")
        rsp = gaussian_response(wl, center=center, fwhm=fwhm)

        if normalize:
            area = np.trapezoid(rsp, wl)
            if area <= 0:
                raise ValueError("Cannot normalize responses with non-positive area.")
            rsp = rsp / area

        return cls(
            name=name,
            wavelengths=wl,
            responses=rsp,
            definition_type="center_fwhm",
            center=float(center),
            fwhm=float(fwhm),
            wavelength_unit=wavelength_unit,
            normalized=normalize,
            provenance=provenance,
            metadata=metadata or {},
        )

    def plot(self, ax=None, show=True, label=None, xlim=None, **kwargs):
        """
        Plot the spectral response function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure and axis are created.
        show : bool
            Whether to call plt.show() if a new figure is created.
        label : str, optional
            Label for the plot (defaults to band name).
        **kwargs
            Additional keyword arguments passed to plt.plot()

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis with the plotted SRF.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if xlim is not None:
            ax.set_xlim(xlim)

        label = label or self.name

        ax.plot(self.wavelengths, self.responses, label=label, **kwargs)

        ax.set_xlabel(f"Wavelength ({self.wavelength_unit})")
        ax.set_ylabel("Response")
        ax.set_title(f"SRF: {self.name}")

        if label:
            ax.legend()

        if show and ax is not None:
            plt.show()

        return ax
    
    def copy(self) -> BandSRF:
        """Return a deep-ish copy of the band object."""
        return BandSRF(
            name=self.name,
            wavelengths=self.wavelengths.copy(),
            responses=self.responses.copy(),
            definition_type=self.definition_type,
            center=self.center,
            fwhm=self.fwhm,
            wavelength_unit=self.wavelength_unit,
            normalized=self.normalized,
            provenance=self.provenance,
            metadata=dict(self.metadata),
        )

    def area(self) -> float:
        """Return the integrated area under the response curve."""
        return float(np.trapezoid(self.responses, self.wavelengths))

    def normalized_copy(self) -> BandSRF:
        """Return a normalized copy of the band response."""
        area = self.area()
        if area <= 0:
            raise ValueError("Cannot normalize a response with non-positive area.")

        new_band = self.copy()
        new_band.responses = new_band.responses / area
        new_band.normalized = True
        return new_band

    def interpolate_to(self, new_wavelengths: ArrayLike1D) -> BandSRF:
        """
        Interpolate the band response onto a new wavelength grid.

        Values outside the original wavelength range are set to 0.
        """
        new_wl = _as_1d_float_array(new_wavelengths, "new_wavelengths")
        _validate_strictly_increasing(new_wl, "new_wavelengths")

        new_rsp = np.interp(
            new_wl,
            self.wavelengths,
            self.responses,
            left=0.0,
            right=0.0,
        )

        return BandSRF(
            name=self.name,
            wavelengths=new_wl,
            responses=new_rsp,
            definition_type=self.definition_type,
            center=self.center,
            fwhm=self.fwhm,
            wavelength_unit=self.wavelength_unit,
            normalized=self.normalized,
            provenance=self.provenance,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class SensorSRF:
    """
    Collection of BandSRF objects representing one sensor.

    Attributes
    ----------
    name : str
        Sensor name, e.g. 'S2B' or 'AltumPT'.
    bands : dict[str, BandSRF]
        Mapping of band name to BandSRF.
    wavelength_unit : str
        Common wavelength unit for the sensor.
    provenance : str | None
        Optional description of the sensor-level source.
    metadata : dict[str, Any]
        Free-form metadata.
    """

    name: str
    bands: dict[str, BandSRF] = field(default_factory=dict)
    wavelength_unit: str = "nm"
    provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for band_name, band in self.bands.items():
            if not isinstance(band, BandSRF):
                raise TypeError(
                    f"All entries in bands must be BandSRF objects. "
                    f"Key '{band_name}' has type {type(band).__name__}."
                )
            if band_name != band.name:
                raise ValueError(
                    f"Band dictionary key '{band_name}' does not match BandSRF.name '{band.name}'."
                )
            if band.wavelength_unit != self.wavelength_unit:
                raise ValueError(
                    f"Band '{band.name}' wavelength_unit='{band.wavelength_unit}' does not match "
                    f"sensor wavelength_unit='{self.wavelength_unit}'."
                )

    def add_band(self, band: BandSRF, overwrite: bool = False) -> None:
        """Add a BandSRF to the sensor."""
        if band.wavelength_unit != self.wavelength_unit:
            raise ValueError(
                f"Band wavelength_unit '{band.wavelength_unit}' does not match "
                f"sensor wavelength_unit '{self.wavelength_unit}'."
            )

        if band.name in self.bands and not overwrite:
            raise ValueError(
                f"Band '{band.name}' already exists. Use overwrite=True to replace it."
            )

        self.bands[band.name] = band

    def get_band(self, name: str) -> BandSRF:
        """Get a band by name."""
        try:
            return self.bands[name]
        except KeyError as exc:
            available = ", ".join(self.band_names()) or "(none)"
            raise KeyError(f"Band '{name}' not found. Available bands: {available}") from exc

    def band_names(self) -> list[str]:
        """Return sorted band names."""
        return sorted(self.bands.keys())

    def copy(self) -> SensorSRF:
        """Return a copy of the SensorSRF."""
        return SensorSRF(
            name=self.name,
            bands={name: band.copy() for name, band in self.bands.items()},
            wavelength_unit=self.wavelength_unit,
            provenance=self.provenance,
            metadata=dict(self.metadata),
        )

    @property
    def source_type(self) -> str:
        """
        Describe how the sensor bands were defined.

        Returns
        -------
        str
            'explicit_curve', 'center_fwhm', or 'mixed'
        """
        if not self.bands:
            return "empty"

        types = {band.definition_type for band in self.bands.values()}
        if len(types) == 1:
            return next(iter(types))
        return "mixed"

    def normalize_bands(self, inplace: bool = False) -> SensorSRF:
        """
        Normalize all band response curves by area.

        Parameters
        ----------
        inplace : bool
            If True, modify this object in place. Otherwise return a new object.

        Returns
        -------
        SensorSRF
            Normalized sensor object.
        """
        target = self if inplace else self.copy()

        for name, band in list(target.bands.items()):
            target.bands[name] = band.normalized_copy()

        return target

    def interpolate_bands_to(
        self,
        new_wavelengths: ArrayLike1D,
        inplace: bool = False,
    ) -> SensorSRF:
        """
        Interpolate all bands onto a new wavelength grid.
        """
        target = self if inplace else self.copy()

        for name, band in list(target.bands.items()):
            target.bands[name] = band.interpolate_to(new_wavelengths)

        return target
