#!/usr/bin/env python3
"""
Given an eROSITA SRCTOOL light curve,
Computes a Bayesian excess variance, by estimating mean and variance of
the log of the count rate.

The model allows a different source count rate in each time bin,
but marginalises over this.
First the source count rate PDF is determined in each time bin,
on a fixed source count grid. The nested sampling algorithm
explores mean and variance combinations and
combines the source count rate PDFs with the mean/variance
information.
No time information is used (order is irrelevant).

Run as:

$ python bexvar_ero.py 020_LightCurve_00001.fits

It will make a few visualisations and a file containing the resulting parameters.

* 020_LightCurve_00001.fits-bexvar-<band number>-corner.png:

  * plot of the intrinsic source count rate and its excess variance

* 020_LightCurve_00001.fits-bexvar-<band number>.png:

  * plot of the light curve and estimated intrinsic rates

* 020_LightCurve_00001.fits-bexvar-<band number>.fits:

  * Bayesian light curve rates in table, with columns

    * 'TIME': time (from SRCTOOL light curve)
    * 'RATE': source count rate
    * 'RATE_LO': source count rate, lower 1 sigma quantile
    * 'RATE_HI': source count rate, upper 1 sigma quantile

  * header:

    * 'pvar_p1': probability that the excess variance exceeds 0.1 dex,
    * 'pvar_p3': probability that the excess variance exceeds 0.3 dex,
    * 'pvar_1': probability that the excess variance exceeds 1 dex,
    * 'rate': estimated mean log(count rate),
    * 'rate_err': uncertainty of the mean log(count rate),
    * 'scatt': estimated scatter of the log(count rate) in dex,
    * 'scatt_lo': lower 1 sigma quantile of the estimated scatter of the log(count rate) in dex,
    * 'scatt_hi': upper 1 sigma quantile of the estimated scatter of the log(count rate) in dex,


Authors: Johannes Buchner, David Bogensberger

"""

import matplotlib.pyplot as plt
from numpy import log
import numpy as np
import scipy.stats, scipy.optimize
import sys
from astropy.table import Table

__version__ = '1.1.1'
__author__ = 'Johannes Buchner'

# 1-sigma quantiles and median
quantiles = scipy.stats.norm().cdf([-1, 0, 1])

N = 1000
M = 1000

def lscg_gen(src_counts, bkg_counts, bkg_area, rate_conversion, density_gp):
    """
    Generates a log_src_crs_grid applicable to this particular light curve,
    with appropriately designated limits, for a faster and more accurate
    run of estimate_source_cr_marginalised and bexvar
    """
    # lowest count rate
    a = scipy.special.gammaincinv(src_counts + 1, 0.001) / rate_conversion
    # highest background count rate
    b = scipy.special.gammaincinv(bkg_counts + 1, 0.999) / (rate_conversion * bkg_area)
    mindiff = min(a - b)
    if mindiff > 0: # background-subtracted rate is positive
        m0 = np.log10(mindiff)
    else: # more background than source -> subtraction negative somewhere
        m0 = -1
    # highest count rate (including background)
    c = scipy.special.gammaincinv(src_counts + 1, 0.999) / rate_conversion
    m1 = np.log10(c.max())
    # print(src_counts, bkg_counts, a, b, m0, m1)

    # add a bit of padding to the bottom and top
    lo = m0 - 0.05 * (m1 - m0)
    hi = m1 + 0.05 * (m1 - m0)
    span = hi - lo
    if lo < -1:
        log_src_crs_grid = np.linspace(-1.0, hi, int(np.ceil(density_gp * (hi + 1.0))))
    else:
        log_src_crs_grid = np.linspace(lo, hi, int(np.ceil(density_gp * 1.05 * span)))

    return log_src_crs_grid

def estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, rate_conversion):
    """ Compute the PDF at positions in log(source count rate)s grid log_src_crs_grid
    for observing src_counts counts in the source region of size src_area,
    and bkg_counts counts in the background region of size bkg_area.

    """
    # background counts give background cr deterministically
    u = np.linspace(0, 1, N)[1:-1]
    bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) / bkg_area
    def prob(log_src_cr):
        src_cr = 10**log_src_cr * rate_conversion
        like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
        return like

    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])
    if not weights.sum() > 0:
        print("WARNING: Weight problem! sum is", weights.sum(), np.log10(src_counts.max() / rate_conversion), log_src_crs_grid[0], log_src_crs_grid[-1])
    weights /= weights.sum()

    return weights

def bexvar(log_src_crs_grid, pdfs):
    """
    Assumes that the source count rate is log-normal distributed.
    returns posterior samples of the mean and std of that distribution.

    pdfs: PDFs for each object
          defined over the log-source count rate grid log_src_crs_grid.

    returns (log_mean, log_std), each an array of posterior samples.
    """

    def transform(cube):
        params = cube.copy()
        params[0] = cube[0] * (log_src_crs_grid[-1] - log_src_crs_grid[0]) + log_src_crs_grid[0]
        params[1] = 10**(cube[1]*4 - 2)
        return params

    def loglike(params):
        log_mean  = params[0]
        log_sigma = params[1]
        # compute for each grid log-countrate its probability, according to log_mean, log_sigma
        variance_pdf = scipy.stats.norm.pdf(log_src_crs_grid, log_mean, log_sigma)
        # multiply that probability with the precomputed probabilities (pdfs)
        likes = log((variance_pdf.reshape((1, -1)) * pdfs).mean(axis=1) + 1e-100)
        like = likes.sum()
        if not np.isfinite(like):
            like = -1e300
        return like


    from ultranest import ReactiveNestedSampler
    sampler = ReactiveNestedSampler(['logmean', 'logsigma'], loglike,
        transform=transform, vectorized=False)
    samples = sampler.run(viz_callback=False)['samples']
    sampler.print_results()
    log_mean, log_sigma = samples.transpose()

    return log_mean, log_sigma
