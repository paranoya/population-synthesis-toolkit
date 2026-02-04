"""
Module containing generic utility functions
"""

import numpy as np
from astropy import units as u
from scipy.sparse import csr_matrix
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from pst.model import Parameter

SQRT_2 = np.sqrt(2)

def _const_interp_cumflux(edges_out, edges_in, f_centers_seg):
    """
    Interpolate the cumulative integrated flux assuming constant
    flux density within each input bin.

    The function constructs the cumulative integral
    C(lambda) = integral f(lambda') d(lambda') evaluated at the
    input bin edges and linearly interpolates C onto the output
    edges. Differences of the interpolated cumulative at consecutive
    output edges yield the integrated flux per output bin.
    Because the interpolation operates on the cumulative, total
    flux is conserved by construction.

    Parameters
    ----------
    edges_out : array_like of float, shape (M+1,)
        Target bin edges (must be strictly increasing).
    edges_in : array_like of float, shape (N+1,)
        Input bin edges for the current contiguous segment.
    f_centers_seg : array_like of float, shape (N,)
        Flux density sampled at input bin centers for this segment.

    Returns
    -------
    cum_flux_out : ndarray of float, shape (M+1,)
        Cumulative integrated flux evaluated at edges_out.

    Notes
    -----
    This assumes f is constant within each input bin (zero order hold).
    It is fast and robust and is the default choice for most spectra.
    """
    dlam = np.diff(edges_in)
    cum_flux = np.r_[0.0, np.cumsum(f_centers_seg * dlam)]
    return np.interp(edges_out, edges_in, cum_flux,
                     left=cum_flux[0], right=cum_flux[-1])

def _f_edges_from_centers(f_centers):
    """
    Estimate flux density at bin edges from center samples.
    Intended for use with a piecewise linear model of f(lambda)
    inside each bin.

    Interior edge values are computed as the average of adjacent
    centers. End edges are extrapolated linearly from the nearest
    two centers.

    Parameters
    ----------
    f_centers : array_like of float, shape (N,)
        Flux density at bin centers.

    Returns
    -------
    f_edges : ndarray of float, shape (N+1,)
        Estimated flux density at bin edges.
    """
    n = f_centers.size
    if n == 1:
        # Only one bin: take constant across both edges
        return np.r_[f_centers[0], f_centers[0]]
    f_edges = np.empty(n+1, dtype=f_centers.dtype)
    # interior edges = average of neighbor centers
    f_edges[1:-1] = 0.5 * (f_centers[:-1] + f_centers[1:])
    # end edges by linear extrapolation of centers
    f_edges[0] = f_centers[0] - 0.5 * (f_centers[1] - f_centers[0])
    f_edges[-1] = f_centers[-1] + 0.5 * (f_centers[-1] - f_centers[-2])
    return f_edges

def _lin_interp_cumflux(edges_out, edges_in, f_centers_seg):
    """
    Interpolate the cumulative integrated flux assuming linear
    flux density within each input bin.

    The function estimates f(lambda) at the input edges from
    center values and integrates using the trapezoidal rule to
    obtain the cumulative C(lambda) at the edges. C is then
    linearly interpolated onto the output edges.

    Parameters
    ----------
    edges_out : array_like of float, shape (M+1,)
        Target bin edges.
    edges_in : array_like of float, shape (N+1,)
        Input bin edges for the current segment.
    f_centers_seg : array_like of float, shape (N,)
        Flux density at input bin centers.

    Returns
    -------
    cum_flux_out : ndarray of float, shape (M+1,)
        Cumulative integrated flux at edges_out.
    """
    f_edges = _f_edges_from_centers(f_centers_seg)
    # cumulative integral of f_edges over edges_in
    # cumulative_trapezoid returns length-1, prepend 0
    cum_flux = np.r_[0.0, cumulative_trapezoid(f_edges, edges_in)]
    return np.interp(edges_out, edges_in, cum_flux,
                     left=cum_flux[0], right=cum_flux[-1])

def _spline_interp_cumflux(edges_out, edges_in, f_centers_seg):
    """
    Interpolate the cumulative integrated flux using a monotonic
    cubic (PCHIP) spline.

    The cumulative C(lambda) is built assuming constant f within
    each input bin and is then evaluated on edges_out using a
    monotone cubic interpolant to avoid oscillations and overshoot.

    Parameters
    ----------
    edges_out : array_like of float, shape (M+1,)
        Target bin edges.
    edges_in : array_like of float, shape (N+1,)
        Input bin edges for the current segment.
    f_centers_seg : array_like of float, shape (N,)
        Flux density at input bin centers.

    Returns
    -------
    cum_flux_out : ndarray of float, shape (M+1,)
        Cumulative integrated flux at edges_out.
    """
    dlam = np.diff(edges_in)
    cum_flux = np.r_[0.0, np.cumsum(f_centers_seg * dlam)]
    interpolator = PchipInterpolator(edges_in, cum_flux)
    return interpolator(edges_out)

def _infer_wl_edges_from_centers(wave):
    """
    Infer bin edges from wavelength centers.

    Midpoints between adjacent centers are used for interior edges,
    and the outer edges are extrapolated by half of the end spacing.

    Parameters
    ----------
    wave : array_like of float, shape (N,)
        Monotonic wavelength centers.

    Returns
    -------
    edges : ndarray of float, shape (N+1,)
        Inferred wavelength bin edges.
    """
    wave = np.asarray(wave)
    mid = 0.5*(wave[1:] + wave[:-1])
    left = wave[0] - (mid[0] - wave[0])
    right = wave[-1] + (wave[-1] - mid[-1])
    return np.r_[left, mid, right]

def _segments_from_mask(mask):
    """
    Compute contiguous valid segments from a boolean mask.

    Parameters
    ----------
    mask : array_like of bool, shape (N,)
        True marks valid bins; False marks invalid or gap bins.

    Returns
    -------
    segments : list of tuple
        List of (start, end) index pairs describing valid segments.
    """
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:  starts = np.r_[0, starts]
    if mask[-1]: ends   = np.r_[ends, mask.size]
    return list(zip(starts, ends))

def resample_via_cumulative_masked(
    w_in, f_in, *,
    w_out=None, edges_in=None, edges_out=None,
    mask_valid=None, fill_value=0.0,
    kind="const"
):
    """
    Flux conserving resampling using cumulative integrals.
    Robust to NaN or invalid values by treating them as gaps.

    Each contiguous valid segment builds its own cumulative
    integral C(lambda) on input edges. C is evaluated on the
    output edges using one of the supported interpolation types,
    and the difference between consecutive output edges gives
    the integrated flux. Flux outside all valid segments can
    optionally be filled with a constant value.

    Parameters
    ----------
    w_in : array_like of float
        Input bin centers (monotonic).
    f_in : array_like of float
        Flux density at input centers.
    w_out : array_like of float, optional
        Output bin centers (monotonic). Provide this or edges_out.
    edges_in : array_like of float, optional
        Input bin edges. If None, inferred from w_in.
    edges_out : array_like of float, optional
        Output bin edges. If None, inferred from w_out.
    mask_valid : array_like of bool, optional
        Mask of valid bins. If None, uses finite values of f_in.
    fill_value : float, default 0.0
        Flux density assumed outside valid coverage.
    kind : {"const", "linear", "cubic"}, default "const"
        Interpolation scheme for cumulative inside segments.

    Returns
    -------
    f_out : ndarray of float
        Flux density per unit wavelength on the output grid.
    """
    w_in = np.asarray(w_in)
    f_in = np.asarray(f_in)

    interp = {"const": _const_interp_cumflux,
              "linear": _lin_interp_cumflux,
              "cubic": _spline_interp_cumflux}.get(kind)
    if interp is None:
        raise ValueError("Unknown interpolation method.")

    # Compute wavelength bin edges
    if edges_in is None:
        edges_in = _infer_wl_edges_from_centers(w_in)
    if edges_out is None:
        if w_out is None:
            raise ValueError("Provide w_out or edges_out.")
        edges_out = _infer_wl_edges_from_centers(w_out)

    # Valid mask: finite and not flagged
    if mask_valid is None:
        mask_valid = np.isfinite(f_in)
    else:
        mask_valid = mask_valid & np.isfinite(f_in)

    # dlam_in: computed by the interpolator
    dlam_out = np.diff(edges_out)
    segs = _segments_from_mask(mask_valid)

    F_out = np.zeros_like(dlam_out)
    covered = np.zeros_like(dlam_out)
    # Sum contributions from each valid segment
    for s, e in segs:
        # Segment edges and bins
        ed_seg = edges_in[s:e+1]
        eL, eR = ed_seg[0], ed_seg[-1]
        f_seg  = f_in[s:e]
        xq = np.clip(edges_out, eL, eR)
        # Interpolate the cumulative flux
        cum_q = interp(xq, ed_seg, f_seg)
        # Integrated flux from this segment
        F_out += np.diff(cum_q)

        # Compute total covered length by valid segments per out-bin
        # (equivalent to summing overlaps of each segment with each out bin)
        left  = np.maximum(edges_out[:-1], edges_in[s])
        right = np.minimum(edges_out[1:],  edges_in[e])
        covered += np.clip(right - left, 0.0, None)

    outside = np.clip(dlam_out - covered, 0.0, None)
    fill = np.where(outside > 0, fill_value * outside, 0.0)
    F_out += fill

    # Convert back to flux density
    with np.errstate(divide='ignore', invalid='ignore'):
        f_out = np.where(dlam_out > 0, F_out / dlam_out, 0.0)

    return f_out

def resample_via_bin_frac(
    w_in, f_in, w_out=None, edges_in=None, edges_out=None,
    var_in=None, fill_value=0.0, return_matrix=False
):
    """
    Flux-conserving resampling via bin-overlap fractions (matrix method).

    Builds a sparse mapping matrix between input and output bins whose
    entries are the fractional overlaps (in wavelength) of each input bin
    with each output bin. Optionally propagates variances under the
    assumption of diagonal input covariance.

    Parameters
    ----------
    w_in : (N,) array_like
        Input bin centers (strictly monotonic).
    f_in : (N,) array_like
        Input flux density per unit wavelength.
    w_out : (M,) array_like, optional
        Output bin centers. Provide this or `edges_out`.
    edges_in : (N+1,) array_like, optional
        Input edges. If None, inferred from `w_in`.
    edges_out : (M+1,) array_like, optional
        Output edges. If None, inferred from `w_out`.
    var_in : (N,) array_like, optional
        Input variances of flux density per input bin (uncorrelated).
    fill_value : float, default 0.0
        Flux *density* to add in output bins lying outside the input
        support (e.g., 0.0 for no extrapolation).
    return_matrix : bool, default False
        If True, also return the CSR sparse mapping matrix (N_out, N_in)
        of fractional overlaps.

    Returns
    -------
    f_out : (M,) ndarray
        Output flux density per unit wavelength.
    var_out : (M,) ndarray or None
        Propagated variance (None if `var_in` is None).
    sparse_matrix : scipy.sparse.csr_matrix, optional
        Mapping matrix such that `F_out = M @ (f_in * delta_lambda_in)`.

    Notes
    -----
    - Exactly flux-conserving (up to FP tol.), independent of within-bin
      shape assumptions.
    - Useful for variance propagation and covariance modeling.
    - Complexity O(N_overlaps); efficient with CSR sparsity.
    """
    w_in = np.asarray(w_in)
    f_in = np.asarray(f_in)

    # Edges
    if edges_in is None:
        edges_in = _infer_wl_edges_from_centers(w_in)
    if edges_out is None:
        if w_out is None:
            raise ValueError("Provide either w_out or edges_out.")
        edges_out = _infer_wl_edges_from_centers(w_out)
    else:
        edges_out = np.asarray(edges_out)

    # Filter NaNs/Infs
    finite = np.isfinite(f_in)
    if var_in is not None:
        var_in = np.asarray(var_in)
        finite &= np.isfinite(var_in)
    if not np.all(finite):
        f_in = np.where(finite, f_in, 0.0)
        if var_in is not None:
            var_in = np.where(np.isfinite(var_in), var_in, 0.0)

    dlam_in  = np.diff(edges_in)
    dlam_out = np.diff(edges_out)

    # Find overlapping bins
    iL = np.searchsorted(edges_in, edges_out[:-1], side="right") - 1
    iR = np.searchsorted(edges_in, edges_out[1:],  side="left")  - 1
    iL = np.clip(iL, 0, dlam_in.size - 1)
    iR = np.clip(iR, 0, dlam_in.size - 1)

    # Number of input bins intersecting with each output bin
    runlen = np.maximum(0, iR - iL + 1)

    # Build a compressed sparse row (CSR) matrix (N_out, N_in)
    # containing the fraction of flux from each input resolution element.
    indptr = np.r_[0, np.cumsum(runlen)]
    nnz    = int(indptr[-1])

    rows = np.repeat(np.arange(dlam_out.size), runlen)
    offsets = np.arange(nnz) - np.repeat(indptr[:-1], runlen)
    cols = np.repeat(iL, runlen) + offsets

    # Overlap lengths for every (row, col) pair
    left_out = np.repeat(edges_out[:-1], runlen)
    right_out = np.repeat(edges_out[1:],  runlen)
    left_in  = edges_in[cols]
    right_in  = edges_in[cols + 1]
    overlap = np.minimum(right_out, right_in) - np.maximum(left_out, left_in)
    overlap /= dlam_in[cols]

    keep = overlap > 0.0
    rows, cols, data = rows[keep], cols[keep], overlap[keep]

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(dlam_out.size, dlam_in.size))

    # --- Apply mapping on integrated flux ---
    F_in  = f_in * dlam_in
    F_out = sparse_matrix.dot(F_in)

    # Only add fill outside input support
    inside = (edges_out[:-1] >= edges_in[0]) & (edges_out[1:] <= edges_in[-1])
    if fill_value != 0.0:
        row_sums = np.array(sparse_matrix.sum(axis=1)).ravel()
        outside  = np.where(inside, 0.0, np.clip(dlam_out - row_sums, 0.0, None))
        F_out   += fill_value * outside

    with np.errstate(divide='ignore', invalid='ignore'):
        f_out = np.where(dlam_out > 0, F_out / dlam_out, np.nan)

    # Variance (diagonal input covariance)
    var_out = None
    if var_in is not None:
        var_Fin  = var_in * (dlam_in**2)
        var_Fout = sparse_matrix.power(2.0).dot(var_Fin)
        with np.errstate(divide='ignore', invalid='ignore'):
            var_out = np.where(dlam_out > 0, var_Fout / (dlam_out**2), 0.0)

    if return_matrix:
        return f_out, var_out, sparse_matrix
    else:
        return f_out, var_out


def flux_conserving_interpolation(new_wave, wave, spectra, *,
                                  method="binfrac",
                                  spectra_err=None, **interp_args):
    """
    High-level wrapper for flux-conserving spectral resampling.

    Provides two flux-conserving resampling paths:
    (i) cumulative piecewise method robust to gaps/outliers (`method="cumulative"`)
        via `resample_via_cumulative_masked`, and
    (ii) exact bin-overlap matrix method (`method="binfrac"`) via
        `resample_via_bin_frac` with optional variance propagation.

    Parameters
    ----------
    new_wave : array_like or Quantity
        Target wavelength centers. If Quantity, units are respected; if not,
        treated as unitless but must be consistent with `wave`.
    wave : array_like or Quantity
        Input wavelength centers.
    spectra : array_like or Quantity
        Flux density at `wave`. If Quantity, units are preserved.
    method : {"cumulative","binfrac"}, default "cumulative"
        Resampling engine. See functions referenced above for details and
        additional behavior (gaps, variance, matrix return).
    spectra_err : array_like or Quantity, optional
        1-sigma uncertainty on `spectra`. If provided with `method="binfrac"`,
        variances are propagated. For `method="cumulative"`, `spectra_err`
        is resampled as a field (no covariance coupling).
    **interp_args
        Passed through to the selected engine:
        - cumulative: `w_out`, `edges_in`, `edges_out`, `mask_valid`,
          `fill_value`, `kind`
        - binfrac: `w_out`, `edges_in`, `edges_out`, `fill_value`,
          `return_matrix` (ignored here)

    Returns
    -------
    out : Quantity or tuple
        If `spectra_err` is None: resampled spectrum as Quantity with the
        same unit as `spectra`. If `spectra_err` is given: a tuple
        `(spectrum, variance)` where both are Quantity, and variance has
        squared units of `spectra`.

    Notes
    -----
    - For rigorous covariance propagation, prefer `method="binfrac"`.
    """
    if isinstance(wave, u.Quantity):
        new_wave = check_unit(new_wave, wave.unit)
        wl_unit = wave.unit
        wl = wave.value
        new_wl = new_wave.value
    elif isinstance(new_wave, u.Quantity):
        raise ValueError("New wavelength is a Quantity but original wavelength"
                         " is not")
    else:
        wl_unit = 1.0
        wl = wave
        new_wl = new_wave

    if isinstance(spectra, u.Quantity):
        spec_unit = spectra.unit
        spec = spectra.value
    else:
        spec_unit = 1.0
        spec = spectra

    if spectra_err is not None:
            if isinstance(spectra_err, u.Quantity):
                spectra_var_v = spectra_err.to_value(spec_unit)**2
            else:
                spectra_var_v = spectra_err**2
    else:
        spectra_var_v = None
    # Select interpolation method
    if method == "cumulative":
        int_spec = resample_via_cumulative_masked(wl, spec, w_out=new_wl,
                                                  **interp_args)
        int_var = None
        if spectra_err is not None:
            int_var = resample_via_cumulative_masked(wl, spectra_var_v,
                                                     w_out=new_wl,
                                                     **interp_args)
    elif method == "binfrac":
        int_spec, int_var = resample_via_bin_frac(wl, spec,
                                                  w_out=new_wl,
                                                  var_in=spectra_var_v,
                                                  **interp_args)
    else:
        raise KeyError(f"Unrecognised interpolation method {method}")
    if spectra_var_v is None:
        return int_spec * spec_unit
    return int_spec * spec_unit, int_var * spec_unit**2

def gaussian1d_conv(f, sigma, deltax):
    """Apply a gaussian convolution to a 1D array f(x).

    params
    ------
    f : np.array
        1D array containing the data to be convolved with.
    sigma : np.array
        1D array containing the values of sigma at each value of x
    deltax : float
        Step size of x in "physical" units.
    """
    sigma_pixels = sigma / deltax
    pix_range = np.arange(0, f.size, 1)
    if len(pix_range) < 2e4:
        XX = pix_range[:, np.newaxis] - pix_range[np.newaxis, :]
        g = np.exp(- (XX)**2 / 2 / sigma_pixels[np.newaxis, :]**2) / (
                   sigma_pixels[np.newaxis, :] * np.sqrt(2 * np.pi))
        g /= g.sum(axis=1)[:, np.newaxis]
        f_convolved = np.sum(f[np.newaxis, :] * g, axis=1)
    else:
        print(' WARNING: TOO LARGE ARRAY --- APPLYING SLOW CONVOLUTION METHOD ---')
        f_convolved = np.zeros_like(f)
        for pixel in pix_range:
            XX = pixel - pix_range
            g = np.exp(- (XX)**2 / 2 / sigma_pixels**2) / (
                       sigma_pixels * np.sqrt(2 * np.pi))
            g /= g.sum()
            f_convolved[pixel] = np.sum(f * g)
    return f_convolved

def check_unit(quantity, default_unit=None, equivalence=None, **equiv_kwargs):
    """Check the units of an input quantity.
    
    Parameters
    ----------
    quantity : np.ndarray or astropy.units.Quantity
        Input quantity.
    default_unit : astropy.units.Quantity, default=None
        If `quantity` has not units, it corresponds to the unit assigned to it.
        Otherwise, it is used to check the equivalency with `quantity`.
    """
    isq = isinstance(quantity, u.Quantity)
    if isq and default_unit is not None:
        if not quantity.unit.is_equivalent(default_unit):
            if equivalence is not None:
                return quantity.to(default_unit,
                equivalencies=equivalence(**equiv_kwargs))
            else:
                raise u.UnitTypeError(
                    "Input quantity does not have the appropriate units")
        else:
            return quantity.to(default_unit)

    elif not isq and default_unit is not None:
        return quantity << default_unit
    elif not isq and default_unit is None:
        raise ValueError("Input value must be a astropy.units.Quantity")
    else:
        return quantity

def check_parameter(quantity):
    """TODO"""
    if not isinstance(quantity):
        return Parameter(quantity)

def broadcast_to_axis(x: np.ndarray, target_ndim: int, axis: int) -> np.ndarray:
    """
    Expand 1D array x (wavelength axis) to match target ndim, placing wavelength on `axis`.
    """
    if x.ndim != 1:
        raise ValueError("Expected 1D array for wavelength-dependent quantity.")
    if target_ndim == 1:
        return x
    # Create shape like (1,1,...,N,...,1) with N at axis
    shape = [1] * target_ndim
    shape[axis] = x.size
    return x.reshape(shape)
