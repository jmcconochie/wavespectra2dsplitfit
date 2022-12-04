=================================
Wave Spectra 2D Splitting/Fitting
=================================

Introduction
============

The main purpose of this package is to find parameters of JONSWAP wave spectra with spreading that, when recombined,
 best match the input 2D frequency direction wave spectra.  Given a 2D wave spectrum S(f,theta), the package
 finds parameters of multiple JONSWAP partitions including wave spreading (i.e. Hs, Tp, Gamma, Tail exponent, ThetaP).  

The aim of the package is to provide an industry wide approach to derive usable wave spectral parameters that
provide the best possible reconstruction of the input wave spectrum.  The method is designed to be tunable, but
robust in the default configuration.  A large number of observed and numerically modelled datasets have been tested 
during the creation and validation of the method.

It is the intention that the package will be used by consultants and weather forecastors to improve the descriptions
of the ocean wave partitions for use in operations and engineering applications.  It provides the metocean engineer
with a robust way to separate swells and wind seas.


Usage
=====

Import the waveSpec class

.. code-block:: python
    from wavespectra2dsplitfit import waveSpec

