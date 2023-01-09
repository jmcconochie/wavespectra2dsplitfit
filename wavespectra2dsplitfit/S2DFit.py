#!/usr/bin/env python
# coding: utf-8

# # General Purpose Functions

# ## angDiff

# In[1]:


def angDiff(fromAngle,toAngle):
    """Calculate the difference in angle [deg] fromAngle relative to toAngle
    
    Args:
        - fromAngle (1darray): vector of from angles in [deg]
        - toAngle (float): to angle in [deg]
    
    Returns:
        - angDiff (1darray): difference in angle fromAngle relative to toAngle
    
    Reference: 
        - Written: Jason McConochie 15/Dec/2022
    
    """
    
    import numpy as np
    ang1 = np.mod(fromAngle,360.)
    ang2 = np.mod(toAngle,360.)
    angDiff1 = ang1-ang2
    m = angDiff1 > 180.
    angDiff1[m] -= 360.
    m = angDiff1 < -180.
    angDiff1[m] += 360.
    return angDiff1  


# # General Purpose Wave Functions

# ## wavenuma

# In[2]:


def wavenuma(freq, water_depth):
    """Chen and Thomson wavenumber approximation.
    
    Args:
        - freq (DataArray, 1darray, float): Frequencies (Hz).
        - water_depth (DataArray, float): Water depth (m).
    Returns:
        - k (DataArray, 1darray, float): Wavenumber 2pi / L.
    
    Reference: 
        - Code taken from https://github.com/oceanum/wavespectra/blob/master/wavespectra/core/utils.py
    
    """
    
    import numpy as np
    ang_freq = 2 * np.pi * freq
    k0h = 0.10194 * ang_freq * ang_freq * water_depth
    D = [0, 0.6522, 0.4622, 0, 0.0864, 0.0675]
    a = 1.0
    for i in range(1, 6):
        a += D[i] * k0h ** i
    return (k0h * (1 + 1.0 / (k0h * a)) ** 0.5) / water_depth


# ## celerity

# In[3]:


def celerity(freq, depth=None):
    """Wave celerity C.
    
    Args:
        - freq (ndarray): Frequencies (Hz) for calculating C.
        - depth (float): Water depth, use deep water approximation by default.
    
    Returns;
        - C: ndarray of same shape as freq with wave celerity (m/s) for each frequency.
    
    Reference: 
        - Code taken from https://github.com/oceanum/wavespectra/blob/master/wavespectra/core/utils.py
    
    """
    
    if depth is not None:
        import numpy as np
        ang_freq = 2 * np.pi * freq
        return ang_freq / wavenuma(freq, depth)
    else:
        return 1.56 / freq  


# # Make Wave Spectrum Functions

# ## dfFromf

# In[4]:


def dfFromf(f):
    """Calculate frequency spacings
    
    Args:
        - f (nparray): Array of frequencies in Hz 
        
    Returns:
        - df (1darray): frequency spacing for each frequency

    """
    import numpy as np
    df = np.zeros(np.size(f))
    df[0] = f[1] - f[0]
    df[1:-1] = ( f[2:] - f[1:-1] ) / 2 + ( f[1:-1] - f[0:-2] ) / 2
    df[-1] = f[-1] - f[-2]
    return df 


# ## dthFromth

# In[5]:


def dthFromth(th):
    """Calculate direction spacings
    
    Args:
        - th (nparray): Array of directions in deg 
        
    Returns:
        - dth (nparray): direction spacing for each direction

    """
    import numpy as np
    dth = np.zeros(np.size(th))
    dth[0] = th[1] - th[0]
    dth[1:-1] = ( th[2:] - th[1:-1] ) / 2 + ( th[1:-1] - th[0:-2] ) / 2
    dth[-1] = th[-1] - th[-2]
    return dth  


# ## JONSWAPCos2s

# In[6]:


def JONSWAPCos2s(f, th, fSpecParm, dSpecParm, spreadType='parametric'):
    """Make 2D Spectrum using JONSWAP and cos2s directional spreading 
    
    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - fSpecParm (nparray): Array containing Hs,Tp,gamma,sigmaa,sigmab,jexp
        - dSpecParm (nparray): Array containing waveDir,s
        - spreadType (string): if 'parametric' s parameter is ignored and will use 
                Ewans (1998) for wind Tp < 9s and Ewans (2001 Wrapped Normal) for Tp > 9s

    Returns:
        - S: (nparray): 2D Spectrum
        - ThetaP: (nparray): peak wave direction [s]
    
    Reference: 
        - Written: Jason McConochie 15/Dec/2022    
        
    """    
    Sf = JONSWAP(f, *fSpecParm)
    S = applyWaveSpreadingFunction(f, th, Sf, dSpecParm, spreadType)
    return S


# ## multiJONSWAPCos2s

# In[7]:


def multiJONSWAPCos2s(f, th, specParm, spreadType='parametric'):
    """Make multiple 2D Spectra using JONSWAP and cos2s directional spreading 
    
    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - specParm (2darray): Array containing vectors of [Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s]
            e.g. [ [Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s], [Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s], ...]
        - spreadType (string): if 'parametric' s parameter is ignored and will use 
                Ewans (1998) for wind Tp < 9s and Ewans (2001 Wrapped Normal) for Tp > 9s

    Returns:
        - Stot: (2darray): 2D Spectrum containing sum of all partitions
        - Sparts: (3darray): 2D Spectrum containing each partition
    
    Reference: 
        - Written: Jason McConochie 15/Dec/2022    
        
    """    
    import numpy as np

    # A. Construct spectra
    nf = np.size(f)
    nth = np.size(th)
    nparts = len(specParm)
    Stot = np.zeros((nf,nth))
    Sparts = np.array([np.zeros_like(Stot)] * nparts)
    for iPart,tPart in enumerate(specParm):
        Sparts[iPart] = JONSWAPCos2s(f, th, tPart[0:6],tPart[6:9],spreadType)  
        Stot = Stot + Sparts[iPart]
    
    return Stot, Sparts


# ## applyWaveSpreadingFunction

# In[8]:


def applyWaveSpreadingFunction(f, th, Sf, dSpecParm, spreadType='parametric', TpSeaSwellCut=9):
    """Apply a wave spreading function to a wave frequency spectrum returning 2D spectrum 
    
    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - Sf (1darray): vector of frequency spectrum S(f) [m^2/Hz]
        - fSpecParm (1darray): Array containing Hs,Tp,gamma,sigmaa,sigmab,jexp
        - dSpecParm (1darray): Array containing waveDir,s
        - spreadType (string): if 'parametric'
            for Tp < TpSeaSwellCut uses Ewans (1998)
            for Tp >= TpSeaSwellCut uses Ewans (2001 Wrapped Normal) 
            (NB. s parameter in dSpecParm is ignored)
        - TpSeaSwellCut (float): Cutoff period, default 9 [s]

    Returns;
        - S: (nparray): 2D Spectrum [m^2/(Hz.deg)]
    
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """    
   
    import numpy as np
    import numpy.matlib as ml

    useEwansCos2s = True
    if (spreadType == 'parametric'):
        waveDir,Tp = dSpecParm
        if Tp < TpSeaSwellCut:
            if useEwansCos2s:
                # Use Ewans 1998
                s = np.zeros(len(f))
                ffp = f * Tp
                for iFreq,tFreq in enumerate(f):
                    if ffp[iFreq] < 1:
                        s[iFreq] = 15.5*ffp[iFreq]**9.47
                    else:
                        s[iFreq] = 13.1*ffp[iFreq]**-1.94
                Dth = cos2s(th,waveDir,s)
                S = np.zeros((len(f),len(th)))
                for i,f in enumerate(f):
                    S[i,:] = Sf[i] * Dth[i,:] * np.pi/180
            else:
                # Use Ewans with bifurcation
                o = MauiBimodalWN(f,1/Tp)
                dm1 = (waveDir-o[:,0]/2) % 360
                dm2 = (waveDir+o[:,0]/2) % 360
                m1 = dm1 < 360;  dm1[m1] = dm1[m1] + 360
                m2 = dm2 > 360;  dm2[m2] = dm2[m2] - 360
                s = o[:,2]
                n = 5
                Dth = EwansDoubleWrappedNormal(th,dm1,dm2,s,n)
                S = np.transpose(ml.repmat(Sf,len(th),1)) * Dth #* np.pi/180
        else:
            # User Ewans 2001 swell
            Dth = EwansWrappedNormal(f,th,waveDir,Tp)
            S = np.transpose(ml.repmat(Sf,len(th),1)) * Dth * np.pi/180
    else: 
        waveDir,s = dSpecParm
        Dth = cos2s(th,waveDir,s)
        S = np.zeros((len(f),len(th)))
        for i,f in enumerate(f):
            S[i] = Sf[i] * Dth * np.pi/180
    return S


# ## Frequency spectra

# ### JONSWAP

# In[9]:


def JONSWAP(f, Hs, Tp, gamma = 3.3, sigmaa=0.07, sigmab=0.09, tailexp=-5):
    """Create a 1D JONSWAP frequency spectrum 
    
    Args:
        - f (1darray): vector of frequencies [Hz]
        - Hs (float): Significant wave height [m]
        - Tp (float): Peak wave period [s]
        - gamma (float): JONSWAP peakedness parameter
        - sigmaa (float): JONSWAP left spectral width
        - sigmab (float): JONSWAP left spectral width
        - tailexp (float): JONSWAP tail exponent

    Returns;
        - Sf: (1darray): 1D Spectrum [m^2/Hz]
    
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """    
   
    import numpy as np
    g = 9.81      # [m/s^2]
    fp = 1 / Tp   # [Hz]
    sigma = sigmaa * (f < fp) + sigmab * (f >= fp) 
    G = gamma ** np.exp( -1 * ( ((f-fp) ** 2) / ( 2 * sigma**2 * fp**2)) )
    alpha = 1 # will be rescaled
    Sf = alpha * g**2 * (2.*np.pi)**(tailexp+1) * f**tailexp * np.exp( tailexp/4 * (f/fp)**-4 ) * G # m^2/Hz
    calcHs = 4 * np.sqrt(np.trapz(Sf,f))
    if calcHs == 0.0:
        Sf = Sf 
    else:
        Sf = Sf * (Hs / calcHs)**2
    
    return Sf


# ## Wave spreading functions

# ### cos2s

# In[10]:


def cos2s(th,waveDir,s):
    """cos2s 1D or 2D normalised direction
    
    Returns normalised cos2s spread function for each direction
        in the waveSpec instance th [deg] array.
    In case of frequency dependent spreading this is size of 2D Spectrum.
    
    Args:
        - th (1darray): vector of directions [deg]
        - waveDir (float): direction of peak [deg]
        - s (1darray, float): spreading factor 
    
    Returns;
        - Dth (1darray, 2darray): cos2s [1/rad]

    Reference: 
        - Written: Jason McConochie 15/Dec/2022
    
    TODO: Add reference
    """

    import numpy as np
    import scipy.special as ss

    # A. Initialisation
    d2r = np.pi/180.0
    dDirFac = 0.5 * np.abs(angDiff(th,waveDir)) * d2r

    # B. Cos2s function
    def dth(s):
        A = (2.0 ** (2.0*s-1.0)*ss.gamma(s+1.0)**2.0) / (np.pi * ss.gamma(2.0*s+1.0))
        Dth = A * ( np.cos( dDirFac ) ** (2.0 * s) )
        return Dth

    # C. Make Dth - either frequency dependent or not
    if np.shape(s) != ():
        # frequency dependent s
        sg1 = ss.gamma(s+1.0)
        sg2 = ss.gamma(2.0*s+1.0)
        A = (2.0 ** (2.0*s-1.0) * sg1**2.0) / (np.pi * sg2)
        Dth = np.zeros([len(s),len(th)]) 
        for i, ts in enumerate(s):
            Dth[i,:] = A[i] * ( np.cos( dDirFac ) ** (2.0 * s[i]) )
    else:
        Dth = dth(s)

    return Dth    


# ### EwansWrappedNormal

# In[11]:


def EwansWrappedNormal(f, th, waveDir,Tp):
    """Ewans wrapped normal wave spreading distribution
    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - ThetaP (float): wave direction [deg]
        - Tp (float): peak wave period [s]
    Returns;
        - y (ndarray): normalised waves spreading matrix [1/deg]
    Reference: 
        - Ewans ...
    
    TODO: Add reference
    TODO: This should use gmAngDiff - see code
    """

    import numpy as np

    # A. Input conversions
    x0 = waveDir
    x = th
    fp = 1 / Tp
    d2r = np.pi/180.
    d = x * d2r
    d0 = x0 * d2r
    # TODO: This should use gmAngDiff
    delx = np.diff(x) * d2r   # this needs to be fixed
    delx = np.append(delx[1],delx)

    # B. Pre-initialisation
    yo = np.zeros([len(f),len(th)])     # Output matrix
    sqpi = np.sqrt(2*np.pi)
    y0 = np.zeros(len(x))
    y1 = np.ones(len(x))*1/(2*np.pi)
    ffp = f / fp
    a = 6; b = 4; c = -5;         sigma_wn_low = ( a + b * ffp ** c ) * d2r
    a = -36; b = 46; c = 0.3;     sigma_wn_high = ( a + b * ffp ** c ) * d2r
    s = sigma_wn_low * (ffp < 1.0) + sigma_wn_high * (ffp >= 1.0)

    # C. Loop over and do each frequency
    for iFreq,f in enumerate(f):
        # C1. Apply spreading functional form
        if s[iFreq] < 1:
            y = y0
            for i in range(-1,2):
                y = y + np.exp(-1/2*((d-d0-2*np.pi*i)/s[iFreq])**2)/(sqpi*s[iFreq])
        else:
            y = y1
            for i in range(1,6):
                y = y + np.exp(-i*i*s[iFreq]*s[iFreq]/2)/np.pi*np.cos(i*d-i*d0)

        # C3. Make density
        sumYdelX = np.sum(y*delx)
        yo[iFreq,:] = y / sumYdelX   

    return yo


# # General 2D Spectrum Functions

# ## readspec_mat

# In[12]:


def readWaveSpectrum_mat(filename, dates="td", freq="fd", dirn="thetad", spec2d="spec2d"):  
    """Read wave spectra from a matlab file
        
    Variables in matlan file should be:
    - td[nTimes] - vector of matlab date serials
    - fd[nFre] - vector of wave frequencies in Hz
    - thetad[nDir] - vector of wave directions in degrees
    - spec2d[nTimes,nFre,nDir] - array of 2D wave spectra for each time in m^2/(Hz.deg)

    Args:
        - filename (string): name of matlab file
        - dates (string): name of the date serial variable
        - freq (string): name of the frequency variable
        - dirn (string): name of the direction variable
        - spec2d (string): name of the spectrum variable
        
    Returns:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - S (2darray): Array of spectral densities in [m^/(Hz.deg)]
        - sDate (1darray): datetime vector
        
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    import numpy as np
    import scipy.io
    mat = scipy.io.loadmat(filename)
    mat.keys()
    tm = mat[dates]
    f = mat[freq]
    th = mat[dirn]
    S = mat[spec2d] 
    
    import datetime as dt
    sDate = [dt.datetime(x[0],x[1],x[2],x[3],x[4],x[5]) for x in tm]

    return f, th, S, sDate


# ## interpSpectrum

# In[13]:


def interpSpectrum(f,th,S,f_out,th_out):
    """Interpolate 2D spectrum to new frequencies and directions.

    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - S (2darray): Array of spectral densities in [m^/(Hz.deg)]
        - f_out (2darray): Array of requested frequencies in [Hz] of output spectrum
        - th_out (2darray): Array of requested directions in [deg] of output spectrum
        
    Returns:
        - S_out (2darray): Array of spectral densities in [m^/(Hz.deg)] at f_out, th_out
    
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    
    import numpy as np
    from scipy.interpolate import interp2d
    
    f = f.flatten()
    th = th.flatten()
    
    th1=np.hstack((th-360,th,th+360)).flatten()
    S1=np.transpose(np.hstack((S,S,S)))

    interpolator = interp2d(f,th1,S1)
    S_out = np.transpose(interpolator(f_out,th_out))
    
    return S_out    


# ## freqSpecFrom2D

# In[14]:


def freqSpecFrom2D(f, th, S):
    """Make frequency spectrum from 2D spectrum
    
    Args:
        - f (1darray): Array of frequencies [Hz] 
        - th (1darray): Array of directions [deg]
        - S (2darray): Array of spectral densities in [m^/(Hz.deg)]
        
    Returns:
        - Sf (2darray): frequency spectrum [m^2/Hz]

    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    
    import numpy as np
    
    # A. Make frequency spectrum
    dth = dthFromth(th)
    nFre = len(f)
    Sf = np.zeros(nFre)
    for i,tS in enumerate(S):
        Sf[i] = np.sum(tS*dth)
        
    return Sf       


# ## dirSpecFrom2D

# In[15]:


def dirSpecFrom2D(f, th, S):
    """Make direction spectrum from 2D spectrum
    
    Args:
        - f (1darray): Array of frequencies [Hz] 
        - th (1darray): Array of directions [deg]
        - S (2darray): Array of spectral densities in [m^/(Hz.deg)]
        
    Returns:
        - Sth (2darray): direction spectrum [m^2/deg]

    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    
    import numpy as np
    
    # A. Make frequency spectrum
    df = dfFromf(f)
    nDir = len(th)
    Sth = np.zeros(nDir)
    for i,tS in enumerate(np.transpose(S)):
        Sth[i] = np.sum(tS*df)
        
    return Sth


# ## windSeaSpectrumMask

# In[1]:


def windSeaSpectrumMask(f, th, S, wspd, wdir, dpt, agefac = 1.7, seaExp = 1):
    """Estimates part of spectrum to be wind sea based on wind speed and direction
    
    Takes the spectrum returns a wind sea spectrum only containing wave frequencies and directions
    contained within the region defined as that with wave components with a wave age less then
    agefac in the direction of the wind direction (wdir) and for wind speed (wspd)

    wdir must be relative to N, coming from (same as the wave direction convention used)

    The wind speed compoonent is made broader in direction space by using dirSprd expoenent
    on the cos function of the wind speed component (found to be more used for observations)

    If wdir is not give then the wind direction is estimated from the spectrum as the 
    peak direction of all spectral components with a frequency higher than 0.25 Hz
    Estimates wind direction from the spectrum as the mean direction of all spectral components 
    with a frequency higher than 0.25 Hz
    
    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - S (2darray): Array of spectral densities in [m^/(Hz.deg)]
        - wspd (float): Wind speed in m/s
        - wdir (float): Wind direction in degN, coming from
        - dpt (float): Water depth [m]
        - agefac (float): Age factor
        - dirSprd (float): cos function exponent to broaden wind direction selection in direction space
        
    Returns:
        - windseamask (nparray): True/False array where True is wind sea part of spectrum 

    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    import numpy as np
    import numpy.matlib as ml
    
    # A. Make wind sea mask
    wind_speed_component = agefac * wspd * np.cos( np.pi / 180 * angDiff(th, wdir) ) ** (seaExp)
    wave_celerity = celerity(f, dpt)
    wsMask = np.transpose(ml.repmat(wave_celerity,len(th),1)) <= ml.repmat(wind_speed_component,len(f),1)
    if np.logical_not(np.all(wsMask.flatten())):
        return np.zeros_like(S)==0
    
    return wsMask


# ## findSpectrumPeaks

# In[17]:


def findSpectrumPeaks(f, th, S, floorPercentMaxS = 0.05):
    """Find Peaks of spectrum

    Args:
        - f (1darray): spectrum frequencies [Hz]
        - th (1darray): spectrum directions [deg]
        - S (2darray): Array of spectral densities in [m^/(Hz.deg)]
        - floorPercentMaxS (float): the floor of the peak finding as a proportion of max of S(f,th)
        
    Returns:
        - Tp (1darray): Peak periods of the spectral peaks found
        - ThetaP (1darray): Peak directions of the spectral peaks found
        - iTp (1darray): Index of peak periods of the spectral peaks found
        - iThetaP (1darray): Array of index peak directions of the spectral peaks found
        - S (1darray): Array of spectral densities of the spectral peaks found

    Reference: 
        - Written: Jason McConochie 15/Dec/2022
        
    """
    import numpy as np
    from skimage.feature import peak_local_max
    
    # A. Find local maximum of the spectrum
    coordinates = peak_local_max(S, min_distance=1, threshold_abs=floorPercentMaxS*np.amax(S),exclude_border=False)
    nCoord = len(coordinates)
    Tp = np.zeros(nCoord)
    ThetaP = np.zeros(nCoord)
    Spk = np.zeros(nCoord)
    for i,iPeak in enumerate(coordinates):
        Tp[i] = 1/f[coordinates[i][0]]
        ThetaP[i] = th[coordinates[i][1]]
        Spk[i] = S[coordinates[i][0],coordinates[i][1]]

    return Tp, ThetaP, Spk


# ## windDirFromSpectrum

# In[18]:


def windDirFromSpectrum(f, th, S):
    """Estimate wind direction from the spectrum
    
    Estimates wind direction from the spectrum as the mean direction of all spectral components 
    with a frequency higher than 0.25 Hz
    
    Args:
        - f (nparray): Array of frequencies in Hz of input spectrum
        - th (nparray): Array of directions in deg of input spectrum
        - S (nparray): Array of spectral densities in m^/(Hz.deg) of input spectrum
        
    Returns:
        - windDir (float): Wind direction estimated from the wave spectrum

    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    Sws = S * 0.0
    for i,v in enumerate(f):
        if f[i] < 0.25:
            Sws[i,:] = np.zeros(len(th))
    return ThetaM(f, th, Sws)   


# ## smoothSpectrum

# In[19]:


def smoothSpectrum(
    f,th,S, 
    sigmaFreqHz = 0.01, 
    sigmaDirDeg = 5,
    df_smoothing = 0.001,  
    dth_smoothing = 10,
    ):
    """Smooth 2D spectrum using Gaussian smoothing.

    Args:
        - f (nparray): Array of frequencies in Hz of input spectrum
        - th (nparray): Array of directions in deg of input spectrum
        - S (nparray): Array of spectral densities in m^/(Hz.rad) of input spectrum
        - sigmaFreqHz (double): standard deviation of Gaussian smoother in frequency in Hz
        - sigmaDirDeg (double): standard deviation of Gaussian smoother in direction in deg
        - df_smoothing (double): new frequency resolution of smoothed spectrum in Hz
        - dth_smoothing (double): new direction resolution of smoothed spectrum in deg
        
    Returns:
        - f_sm (nparray): Array of frequencies in Hz of smoothed spectrum
        - th_sm (nparray): Array of directions in deg of smoothed spectrum
        - S_sm (nparray): Array of spectral densities in m^/(Hz.rad) at f_out, th_out
        
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    TODO: Allow returning smoothed spectrum on original resolution

    """

    # A. Get STDEV in image units
    sigmaFreq_imUnits = sigmaFreqHz / df_smoothing
    sigmaDir_imUnits = sigmaDirDeg / dth_smoothing

    # B. Regrid the spectrum and extend the edges by wrapping the directions at each end
    import numpy as np
    f_sm = np.arange(f[0], f[-1], df_smoothing)
    th_sm = np.arange(th[0], th[-1], dth_smoothing)
    S_sm = interpSpectrum(f, th, S, f_sm, th_sm)

    # C. Apply Smoothing to the spectrum
    from skimage import img_as_float
    from skimage.filters import gaussian
    im = img_as_float(S_sm)
    S_sm = gaussian(im, sigma=(sigmaFreq_imUnits, sigmaDir_imUnits), truncate=3.5, channel_axis=2)
     
    # D. Return the smoothed spectrum
    return f_sm, th_sm, S_sm


# # Spectral Fitting Functions

# ## fit2DSpectrum

# In[20]:


def fit2DSpectrum(
    f, th, S,
    maxPartitions = 2,
    useClustering = True,
    useWind = False,
    useFittedWindSea = False,
    useWindSeaInClustering = False,
    wspd = None,
    wdir = None,
    dpt = None,
    spreadType = 'parametric',
    agefac = 1.7,                  # wind sea defn
    seaExp = 1.0,                  # wind sea defn
    sigmaFreqHz = 0.01,            # smoothing stdev 
    sigmaDirDeg = 5,               # smoothing stdev
    df_smoothing = 0.001,          # smoothing resolution 
    dth_smoothing = 10,            # smoothing resolution 
    floorPercentMaxS = 0.05,       # peak finding peak floor  
    maxIterFact = 500,             # max interations nonlinear fitting 
    tolIter = 1e-2,                # interation tolerance nonlinear fitting
    ):  
    
    """Fit 2D spectrum using non-linear fitting
    
    Args:
        - f (nparray): Array of frequencies of input spectrum [Hz]
        - th (nparray): Array of directions of input spectrum [deg]
        - S (nparray): Array of spectral densities in m^/(Hz.deg) of input spectrum.
        - maxPartitions (int): maximum number of partitions to return, default=2, optional
        - useClustering (bool): default=True, optional
            True: Selects up to maxPartitions from the peaks in smoothed spectrum using clustering 
            False: Selects up to maxPartitions of the highest peaks in the smoothed spectrum  
        - useWind (bool): default=False, optional
            True: Make use of wind speed and direction and water depth to determine wind sea spectrum
            False: No wind required
        - useFittedWindSea (bool): default=False, optional
            True: TpSea, ThetaPSea calculated by fitting a spectrum to the spectra in the wind sea masked area
            False: TpSea, ThetaPSea from maximum in smoothed spectrum of all peaks in the wind sea mask area
        - useWindSeaInClustering (bool): default=False, optional
            True: Puts TpSea, ThetaPSea into the clustering and lets it take care of it.
            False: Do not use TpSea, ThetaPSea in clustering but instead ensures a wind sea spectrum
                    is fitted in the final fitting as the first partition fixing TpSea, ThetaPSea    
        - wspd: wind speed in m/s, required with useWind=True
        - wdir: wind direction [degN, from] (or same direction datum as spectrum), if
            not provided wdir is taken as mean direction of the spectrum for S(f>0.25Hz), optional        
        - dpt: water depth in m, required with useWind=True
        - agefac (float): Age factor
        - dirSprd (float): cos function exponent to broaden wind direction selection in direction space
        - sigmaFreqHz (double): standard deviation of Gaussian smoother in frequency in Hz
        - sigmaDirDeg (double): standard deviation of Gaussian smoother in direction in deg
        - df_smoothing (double): new frequency resolution of smoothed spectrum in Hz
        - dth_smoothing (double): new direction resolution of smoothed spectrum in deg
    
    Returns;
        - vPart (2darray): Array containing vectors of [Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s]
            e.g. [ [Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s], [Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s], ...]
        - fitStatus (1darray): Array containing [found a solution, the final rmse error, number of function evaluations]
        - diagOut (1darray): 
            f, th, S, Sf, Sth - the input spectrum
            f_sm, th_sm, S_sm - the smoothed spectrum 
            f_ws, th_ws, S_ws, wsMask - the wind sea spectrum
            Tp_pks, ThetaP_pks, Tp_sel, ThetaP_sel, whichClus - the peaks from peak selection and the selected peaks 
                  and the cluster each peak belongs to
        

    Reference: 
        - Written: Jason McConochie 15/Dec/2022
    
    """

    import numpy as np   
    
    # 0. Initial stuff
    S[np.isnan(S)] = 0
    nPeaksToSelect = maxPartitions # For later use define nPeaksToSelect (reduce if wind sea is used)
    f = f.astype(float)
    th = th.astype(float)
    S = S.astype(float) 
    
    # A. Gaussian smoothing and finding spectral peaks
    f_sm, th_sm, S_sm = smoothSpectrum(f, th, S, sigmaFreqHz, sigmaDirDeg, df_smoothing, dth_smoothing)    
    Tp_pk, ThetaP_pk, S_pk = findSpectrumPeaks(f_sm, th_sm, S_sm, floorPercentMaxS)
    iTp_pk = np.zeros(len(Tp_pk),int)
    iThetaP_pk = np.zeros(len(Tp_pk),int)
    for i in range(0,len(Tp_pk)):
        iTp_pk[i] = np.argmin(np.abs(f - (1/Tp_pk[i])))
        iThetaP_pk[i] = np.argmin(np.abs(angDiff(th, ThetaP_pk[i])))
    
    # B1. Estimate wind sea part of the spectrum
    wsMask = np.zeros_like(S) == 0
    if useWind: 
        if wdir == None: # If no wind direction, then estimate it from the 2D wave spectrum
            wdir =  windDirFromSpectrum(f, th, S)
        wsMask = windSeaSpectrumMask(f, th, S, wspd, wdir, dpt, agefac, seaExp)
        useWind = np.shape(wsMask) != ()
    
    # B2. Get the wind sea Tp/ThetaP
    if useWind:
        if useFittedWindSea:
            Tp_ws, ThetaP_ws, S_ws = fitWindSeaSpectrum(f, th, S * wsMask, spreadType, maxIterFact, tolIter) # Get Wind Sea Tp/ThetaP from nonlinear fitting to wind sea masked area
        else:   
            Tp_ws, ThetaP_ws, S_ws = selectPeakInMaskedSpectrum(Tp_pk, ThetaP_pk, iTp_pk, iThetaP_pk, S_pk, wsMask) # Get Wind Sea Tp/ThetaP from highest Peak
        
        # Remove all peaks in wind sea mask area
        mask = wsMask[iTp_pk, iThetaP_pk]
        Tp_pk = Tp_pk[mask] 
        ThetaP_pk = ThetaP_pk[mask]
        
        useWind = Tp_ws != None
  
    # B4. If using wind sea in clustering add to pks array for consideration
    if useWind:
        if useWindSeaInClustering:
            Tp_pk = np.append(Tp_pk,Tp_ws)
            ThetaP_pk = np.append(ThetaP_pk,ThetaP_ws)
            S_pk = np.append(S_pk,S_ws)
        else:
            # Reduce number of peaks to select since one of them will be the wind sea
            if not useClustering:
                nPeaksToSelect = nPeaksToSelect - 1
                
    # C. Clustering or Peak Selection
    whichClus = None
    if useClustering:
        Tp_sel, ThetaP_sel, idx_sel, useClustering, whichClus = selectPeaksByClustering(Tp_pk, ThetaP_pk, S_pk, nPeaksToSelect)
    else:
        Tp_sel, ThetaP_sel = selectPeaksFromSpectrum(Tp_pk, ThetaP_pk, S_pk, nPeaksToSelect)   

    # D. Build all the inital conditions including the fixed Tp/ThetaP for each partition
    parmActive = []  #[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread] 
    parmStart = []
    if useWind and (not useWindSeaInClustering):  
        parmActive.append([True, Tp_ws, True, 0.07, 0.09, True, ThetaP_ws, Tp_ws])
        parmStart.append([2, Tp_ws, 3.3, 0.07, 0.09, -5, ThetaP_ws, Tp_ws])
    for i in range(0,len(Tp_sel)):
        parmActive.append([True, Tp_sel[i], True, 0.07, 0.09, True,ThetaP_sel[i], Tp_sel[i]])
        parmStart.append([2, Tp_sel[i], 3.3, 0.07, 0.09, -5, ThetaP_sel[i], Tp_sel[i]])

    # E. Run the main fitting of the spectrum
    vPart,fitStatus = fitMultiJONSWAPCos2s(f, th, S, parmActive, parmStart, spreadType, maxIterFact, tolIter)

    # F. Sort the partitions in order of increasing Tp
    partTps = [x[1] for x in vPart]
    sPartTps = np.argsort(partTps)
    nPart = [[]]*len(vPart)
    for iPart,tPart in enumerate(vPart):
        nPart[iPart] = vPart[sPartTps[iPart]] 
    vPart = nPart
    
    diagOut = [ 
        f, th, S, 
        f_sm, th_sm, S_sm, 
        wsMask,
        Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus
    ]

    return vPart, fitStatus, diagOut


# ## fitWindSeaSpectrum

# In[18]:


def fitWindSeaSpectrum(f, th, S, spreadType, maxIterFact=500, tolIter=1e-2):
    """Fit 2D spectrum using non-linear fitting
    
    Args:
        - f (nparray): Array of frequencies in Hz of input spectrum.
        - th (nparray): Array of directions in deg of input spectrum.
        - S (nparray): Array of spectral densities in m^/(Hz.deg) of input spectrum.
        - spreadType (string): 'parametric'
    
    Returns;
        - Tp: (float): peak wave period [s]
        - ThetaP: (float): peak wave direction [s]        

    Reference: 
        - Written: Jason McConochie 15/Dec/2022
    
    """
    import numpy as np
    parmActive=[[True, True, True, 0.07, 0.09, -5, True, True]]
    parmStart=[[2, 6, 3.3, 0.07, 0.09, -5, 180, 6]]
    vWindSea, fitStatusWindSea = fitMultiJONSWAPCos2s(f, th, S, parmActive, parmStart, spreadType, maxIterFact, tolIter)
    
    Tp_ws = vWindSea[0][1]
    ThetaP_ws = vWindSea[0][6] 
    iTp_ws = np.argmin(np.abs(f - (1/Tp_ws)))
    iThetaP_ws = np.argmin(np.abs(angDiff(th, ThetaP_ws)))
    S_ws = S[iTp_ws, iThetaP_ws]
    
    return Tp_ws, ThetaP_ws, S_ws          


# ## selectPeakInMaskedSpectrum

# In[22]:


def selectPeakInMaskedSpectrum(Tp_pk, ThetaP_pk, iTp_pk, iThetaP_pk, S_pk, wsMask):
    """Select the highest peak in the masked spectrum

    Args:
        - f (nparray): Array of frequencies in Hz of input spectrum.
        - th (nparray): Array of directions in deg of input spectrum.
        - S (nparray): Array of spectral densities in m^/(Hz.deg) of input spectrum.
        - spreadType (string): 'parametric'

    Returns;
        - Tp: (float): peak wave period [s]
        - ThetaP: (float): peak wave direction [s]
        
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    import numpy as np
    mask = wsMask[iTp_pk, iThetaP_pk]
    if np.shape(S_pk[mask]) == ():
        return None, None, None
    imax = np.argmax(S_pk[mask])
    return Tp_pk[mask][imax], ThetaP_pk[mask][imax], S_pk[mask][imax]


# ## selectPeaksFromSpectrum

# In[13]:


def selectPeaksFromSpectrum(Tp_pk, ThetaP_pk, S_pk, nPeaksToSelect):
    """Select from list of peaks 
    
    From list of peak period, directions using spectral levels to select limited number of peaks
    
    Args:
        - Tp_pk (1darray): Array of peak periods [s]
        - ThetaP_pk (1darray): Array of peak directions [deg]
        - S_pk (1darray): Array of peak spectral densities in m^/(Hz.deg)
        - nPeaksToSelect (float): number of peaks to choose
    Returns;
        - Tp: (1darray): peak wave period [s]
        - ThetaP: (1darray): peak wave direction [s]     

    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """
    import numpy as np
    
    # Select peaks to keep 
    iS_pk = np.argsort(S_pk)
    nPeaks = np.min([len(iS_pk),nPeaksToSelect])
    Tp_sel = np.zeros(nPeaks)
    ThetaP_sel = np.zeros(nPeaks)
    for i in range(1,nPeaks+1):
        itPk = iS_pk[-i]
        Tp_sel[i-1] = Tp_pk[itPk]
        ThetaP_sel[i-1] = ThetaP_pk[itPk]
    return Tp_sel, ThetaP_sel  


# ## selectPeaksByClustering

# In[24]:


def selectPeaksByClustering(Tp_pk, ThetaP_pk, S_pk, maxPeaks, x1Scale = 2.5):
    """ Uses clustering to select a maximum number of spectral peaks
    
    From list of peak period and directions using spectral levels to select limited number of peaks

    Algorithm will cluster the Tp, ThetaP space into clusters and select, within each
      cluster, one peak.  It takes the peak in each cluster with the largest S(f,th) spectral
      density.  S_pk[nPeaks] should be a vector of the 2D spectral density at each of the Tp, ThetaP pairs.
    
    The normalised clustering space is defined as Tp * cos(ThetaP), x1scale * Tp * sin(ThetaP)
       Default x1scale = 2.5. This helps bring more weight to Tp in the normalised space.

    Args:
        - Tp_pk (1darray): Array of peak periods [s]
        - ThetaP_pk (1darray): Array of peak directions [deg]
        - S_pk (1darray): Array of peak spectral densities in m^/(Hz.deg)
        - maxPeaks (float): maximum number of peaks to choose
        - x1scale - scaling of the y-norm space
    Returns;
        - Tp: (1darray): peak wave period [s]
        - ThetaP: (1darray): peak wave direction [s]     

    Reference: 
        - Written: Jason McConochie 15/Dec/2022
  
    """
    import numpy as np
    import os
    os.environ['OMP_NUM_THREADS'] = str(1)
    from sklearn.cluster import KMeans

    # A. Convert real (Tp, ThetaP) to normalised space
    def real2norm(Tp,ThetaP):
        x = np.zeros([len(Tp),2])
        for i,v in enumerate(Tp):
            x[i,0] = Tp[i] * np.cos(np.pi/180 * ThetaP[i])
            x[i,1] = x1Scale * Tp[i] * np.sin(np.pi/180 * ThetaP[i])
        return x

    # B. Convert normlised space to real space (T, ThetaP)
    def norm2real(x):  # two columns (as returned by real2norm)
        nv = np.shape(x,1)
        Tp = np.zeros(nv)
        ThetaP = np.zeros(nv)
        for i in range(0,nv):
            tTheta = np.arctan2(x[i,1]/x1Scale,x[i,0])
            Tp[i]= x[i,0] / np.cos(tTheta)
            ThetaP[i] = (tTheta * 180/np.pi + 720) % 360
        return Tp,ThetaP

    # C. Kmeans clustering if required
    if len(Tp_pk) > maxPeaks:
        
        # C1. Run kmeans clustering on all peaks 
        features = real2norm(Tp_pk, ThetaP_pk)
        nCl = np.min([len(Tp_pk),maxPeaks])
        kmeans = KMeans(
            init="random",
            n_clusters=nCl,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(features)
        whichClus = kmeans.predict(features)  

        # C2. Select the largest peak in each cluster
        nClusters = len(kmeans.cluster_centers_)
        Tp_sel = np.zeros(nClusters)
        ThetaP_sel = np.zeros(nClusters)
        idx_sel = np.zeros(nClusters,int)
        for tClus in range(0,nClusters):
            idx_pks = np.ndarray.flatten(np.argwhere(tClus == whichClus) )
            tClus_sortedS = np.argsort(S_pk[idx_pks])
            idx_pk = idx_pks[tClus_sortedS[-1]]
            Tp_sel[tClus] = Tp_pk[idx_pk]
            ThetaP_sel[tClus] = ThetaP_pk[idx_pk]
            idx_sel[tClus] = idx_pk
        useClustering = True
     
    else:
        # D. Clustering not required as supplied number of peaks is less than requested
        useClustering = False
        Tp_sel = Tp_pk
        ThetaP_sel = ThetaP_pk
        idx_sel = list(range(0,len(Tp_sel)))
        whichClus = np.zeros_like(Tp_sel)

    return Tp_sel, ThetaP_sel, idx_sel, useClustering, whichClus        


# ## fitMultiJONSWAPCos2s

# In[25]:


def fitMultiJONSWAPCos2s(f, th, S, parmActive, parmStart, spreadType='parametric', maxIterFact=500, tolIter=1e-2):
    """Fit multiple JONSWAP / cos2s frequency direction spectra using non-linear fitting

    Args:
        - f (nparray): Array of frequencies [Hz] 
        - th (nparray): Array of directions [deg]
        - S (nparray): Array of spectral densities [m^/(Hz.deg)]
        - parmActive: [[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread],....]
                provide array of fixed parameters or True (if parameter to be fitted), for each partition
        - parmStart: [[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread],....] 
                provide starting parameters for each partition
        - spreadType (string): 'parametric'
        - maxIterFact (int): Number of iterations factor (default: 500) 
        - tolIter (float): Iteration error tolerance (defaul: 1e-2)

    Returns;
        - allPartParms (2darray): JONSWAP and cos2s wave parameters for each partition
             (e.g. [[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread],...])
        - fitStatus (1darray): 
        
        
    Reference: 
        - Written: Jason McConochie 15/Dec/2022

    """

    # A. Define the spectrum error function
    def specErrF(x,*args):
        import numpy as np

        # A1. Input processing
        f = args[0]
        th = args[1]
        S = args[2]   # this is the input 2D spectrum
        parmActive = args[3] # [[Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s],..]
        
        # A2. Map only active arguments: parmActive - True or FixedValue
        allPartParms = [] # the array of array of each partition parameters with fixed values
                        # set from parmActive and fitted parameters coming from x
        k=0
        for iPart,vPart in enumerate(parmActive):
            tParm = [0,0,0,0,0,0,0,0]
            for iParm,vParm in enumerate(vPart):
                if vParm == True:
                    tParm[iParm] = x[k]
                    k=k+1
                else:
                    tParm[iParm] = vParm
            allPartParms.append(tParm)

        # A3. Make a new spectrum made of all partitions
        nS = S * 0
        for iPart,vPart in enumerate(allPartParms):
            if spreadType == "parametric":
                tS = JONSWAPCos2s(f, th, vPart[0:6], np.array(vPart)[[6,1]], spreadType) 
            else:
                tS = JONSWAPCos2s(f, th, vPart[0:6], vPart[6:9], spreadType) 
            nS = nS + tS

        # A4. Calculate error
        def rmse(predictions, targets):
            return np.sqrt(np.sum(np.power(predictions - targets,2)))
        tErr=rmse(nS,S)

        # A5. Limit parameter ranges
        for iPart,vPart in enumerate(allPartParms):
            q = [
                vPart[0]<0,  # Hs 
                vPart[2]<1,  # gamma 
                vPart[2]>20,  # gamma
                vPart[5]>-1, # tail exponent
                vPart[5]<-50, # tail exponent
                vPart[7]<1,   # spread, s 
                vPart[7]>25   # spread, s 
                ]
            if np.any(q):
                tErr = 1e5

        return tErr

    # B. Map only active arguments: parmStart 
    allParmStart = []
    for iPart,vPart in enumerate(parmActive):
        for iParm,vParm in enumerate(vPart):
            if vParm == True:
                allParmStart.append(parmStart[iPart][iParm])

    maxIter = maxIterFact * len(allParmStart)
    from scipy import optimize
    x = optimize.minimize(specErrF, allParmStart, args=(f, th, S, parmActive), tol=tolIter,
                          method="Nelder-Mead",options={'adaptive':True, 'disp':True, 'maxiter':maxIter})
    
    allPartParms = []
    k=0
    for iPart,vPart in enumerate(parmActive):
        tParm = [0,0,0,0,0,0,0,0]
        for iParm,vParm in enumerate(vPart):
            if vParm == True:
                tParm[iParm] = x.x[k]
                k=k+1
            else:
                tParm[iParm] = vParm
        allPartParms.append(tParm)

    # Append whether if found a solution, the final error, number of function evaluations
    fitStatus = [x['success'], x['fun'], x['nfev'] ]

    return allPartParms, fitStatus


# ## plot2DFittingDiagnostics

# In[26]:


def plot2DFittingDiagnostics(
    specParm, 
    f, th, S, 
    f_sm, th_sm, S_sm, 
    wsMask,
    Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus,
    useWind, useClustering,
    saveFigFilename = 'plot.png',  
    iTime = None,                  
    fTime = None,                
    tag = "S2DFit"                 
    ):
    """
       doPlot: True or False
            If True will make a plot of input spectrum, smoothed spectrum, peaks found and selected and clusters
        saveFigFilename: String filename to save spectrum plot
        plotClusterSpace: True or False
            If True will plot the cluster real and normalized space - used for diagnostics on clustering
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    """
    TODO: Complete these
    """
    # A. Make up total and raw frequency and direction spectrum
    Sf = freqSpecFrom2D(f, th, S)
    Sth = dirSpecFrom2D(f, th, S)
    
    # B. Make up wind sea spectrum
    if useWind: 
        f_ws, th_ws, S_ws = f, th, S * wsMask 
    
    # Make up total reconstructed spectrum, and frequency and direction spectrum
    f_t = f
    th_t = th
    S_t, Sparts_t = multiJONSWAPCos2s(f, th, specParm, spreadType='parametric')
    Sf_t = freqSpecFrom2D(f, th, S_t)
    Sth_t = dirSpecFrom2D(f, th, S_t)
    
    def plotPeaks(ax):
        for i in range(0,len(Tp_pk)):
            ax.plot(1/Tp_pk[i], ThetaP_pk[i], 'w.', ms=16)
            ax.plot(1/Tp_pk[i], ThetaP_pk[i], 'm.', ms=10)
            if useClustering:
                ax.text(1/Tp_sel[i], ThetaP_sel[i], str(whichClus[i]), color='white', fontsize=22, ha="left")
        o = ""
        for i in range(0, len(Tp_sel)):
            ax.plot(1/Tp_sel[i], ThetaP_sel[i], 'w.', ms=16)
            ax.plot(1/Tp_sel[i], ThetaP_sel[i], 'k.', ms=10)
            o += f"{Tp_sel[i]:0.1f}-{ThetaP_sel[i]:0.0f}    "
        ax.text(0, 0, "       "+o, color='w', size=12)

    S[S<1e-9]=1e-9;
    S_sm[S_sm<1e-9]=1e-9;
    
    fg,b = plt.subplots(3,3,figsize=(15,15))
    
    ta = b[0][0]
    ta.pcolormesh(f, th, np.transpose(S))
    ta.pcolormesh(f, th, np.log(np.transpose(S+1e-9)), clim=[-15,0]) 
    if useWind: ta.contour(f_ws, th_ws, np.transpose(wsMask), levels=[0.5], colors='white')
    plotPeaks(ta)
    if fTime != None:
        ta.set_title(f"Input@{fTime},{iTime}")
    else:
        ta.set_title(tag)    
    ta = b[0][1]
    ta.pcolormesh(f_sm, th_sm, np.log(np.transpose(S_sm+1e-9)), clim=[-15,0]) 
    ta.set_title('Smoothed spectrum')
    if useWind: ta.contour(f_ws, th_ws, np.transpose(wsMask), levels=[0.5], colors='white')
    plotPeaks(ta)
    
    ta = b[0][2]
    ta.pcolormesh(f_t, th_t, np.log(np.transpose(S_t+1e-9)), clim=[-15,0]) 
    ta.set_title('Reconstructed spectrum')
    if useWind: ta.contour(f_ws, th_ws, np.transpose(wsMask), levels=[0.5], colors='white')
    plotPeaks(ta)

    ta = b[1][0]
    ta.pcolormesh(f, th, np.transpose(S)) 
    if useWind: ta.contour(f_ws, th_ws, np.transpose(wsMask), levels=[0.5], colors='white')
    plotPeaks(ta)
    
    ta = b[1][1]
    ta.pcolormesh(f_sm, th_sm, np.transpose(S_sm)) 
    ta.set_title('Smoothed spectrum')
    if useWind: ta.contour(f_ws, th_ws, np.transpose(wsMask), levels=[0.5], colors='white')
    plotPeaks(ta)
    
    ta = b[1][2]
    ta.pcolormesh(f_t, th_t, np.transpose(S_t)) 
    ta.set_title('Reconstructed spectrum')
    if useWind: ta.contour(f_ws, th_ws, np.transpose(wsMask), levels=[0.5], colors='white')
    plotPeaks(ta)

    ta = b[2][0]
    ta.plot(f, Sf, 'k-', label="Input")
    ta.plot(f_t, Sf_t, 'b-', label="Reconstructed")
    ta.legend()
    ta.set_title('Frequency spectrum')
    
    ta = b[2][1]
    ta.plot(th, Sth, 'k-', label="Input")
    ta.plot(th_t, Sth_t, 'b-', label="Reconstructed")
    ta.set_title('Direction spectrum')
    
    ta = b[2][2]
    ta.plot(f, Sf, 'k-', label="Input")
    ta.plot(f_t, Sf_t, 'b-', label="Reconstructed")
    ta.set_xscale('log')
    ta.set_yscale('log')
    ta.set_ylim([1e-3, 1e3])
    ta.set_title('Frequency spectrum')
    plt.grid()

    fg.savefig(saveFigFilename+"_pk.png")
    plt.close(fg)


# ## plotClusterSpace

# In[27]:


def plotClusterSpace():
    ####plotClusterSpace - if True will make a plot of the Tp-v-ThetaP space peaks as well
    ####   as a plot of the normalised clustering space.
    ####tag - text string identifier pre-predended to filename of plot image saved to png.
    ###   May include the path (e.g.  c:\myPathtoDir\tagName)
    

    import matplotlib.pyplot as plt
    f,a = plt.subplots(1,2,figsize=(10,5))
    ta = a[0]
    ta.plot(Tp_pk,ThetaP_pk,'k.',ms=16)
    ta.plot(Tp_sel,ThetaP_sel,'m.',ms=8)
    for i in range(0,len(features[:,0])):
        ta.text(Tp_pk[i],ThetaP_pk[i],str(whichClus[i])+f" ({Tp_pk[i]:.1f},{ThetaP_pk[i]:.0f},{S_pk[i]:.4f})",fontsize=10)
    ta.set_xlim([0,20])
    ta.set_ylim([0,360])
    ta.set_title('Real space Tp(x) ThetaP(y)')
    ta = a[1]
    ta.plot(features[:,0],features[:,1],'k.',ms=16)
    ta.plot(features[idx_sel,0],features[idx_sel,1],'m.',ms=8)
    ta.axis('square')
    for i in range(0,len(features[:,0])):
        ta.text(features[i,0],features[i,1],str(whichClus[i])+f" ({Tp_pk[i]:.1f},{ThetaP_pk[i]:0.0f},{S_pk[i]:2.4f})",fontsize=10)
    ta.set_title('Normal/clustering space')
    f.savefig(f"{tag}_cs.png")
    plt.close(f)


# # Make python file

# In[26]:


#!jupyter nbconvert S2DFit.ipynb --to python


# # Testing

# In[29]:


def readspec_mat(filename, dates="td", freq="fd", dirn="thetad", spec2d="spec2d"):  

    import scipy.io
    mat = scipy.io.loadmat(filename)
    mat.keys()
    tm = mat[dates]
    f_in = mat[freq]
    th_in = mat[dirn]
    S_in = mat[spec2d]

    import numpy as np
    import datetime as dt
    sDate = [dt.datetime(x[0],x[1],x[2],x[3],x[4],x[5]) for x in tm]
    f = f_in[0]
    th = th_in[0]
    S = S_in * np.pi/180 # convert from m^2/(Hz.rad) to m^2/(Hz.deg)
    
    return f, th, S, sDate


# In[30]:


def main():
    import numpy as np
    filename = "data/ExampleWaveSpectraObservations.mat"
    f, th, S, sDate = readspec_mat(filename)
    n = np.shape(S)[0]
    for i in range(1):
        print("=========== ",i)
        vPart, fitStatus, diagOut = fit2DSpectrum(f, th, S[i,:,:],maxPartitions=3)
        print(vPart)
        print(fitStatus)

        S_t, Sparts_t = multiJONSWAPCos2s(f, th, vPart, spreadType='parametric')
        import matplotlib.pyplot as plt
        Sparts_t[Sparts_t<1e-6] = 0
        m = np.argmax(Sparts_t,axis=0)
        Sparts_t[Sparts_t<1e-6] = np.nan
        plt.figure()
        plt.pcolormesh(f,th,np.transpose(m))
        plt.colorbar()

        for i in range(0,np.shape(Sparts_t)[0]):
            plt.figure()
            plt.pcolormesh(f, th, np.transpose(Sparts_t[i,:,:]))

        f, th, S, f_sm, th_sm, S_sm, wsMask, Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus = diagOut
        useWind = False
        useClustering = True
        plt.figure()
        plot2DFittingDiagnostics(
            vPart, 
            f, th, S, 
            f_sm, th_sm, S_sm, 
            wsMask,
            Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus,
            useWind, useClustering,
            saveFigFilename = 'plot.png',  
            plotClusterSpace = True,       
            iTime = None,                  
            fTime = None,                
            tag = "S2DFit"  
        )


# In[31]:


#main()


# In[ ]:
































# # Ununsed

# In[32]:


# def _make_df(self):
#     import numpy as np
#     df = np.zeros(np.size(self.f))
#     df[0] = self.f[1] - self.f[0]
#     df[1:-1] = ( self.f[2:] - self.f[1:-1] ) / 2 + ( self.f[1:-1] - self.f[0:-2] ) / 2
#     df[-1] = self.f[-1] - self.f[-2]
#     return df

# def _make_dth(self):
#     import numpy as np
#     dth = np.zeros(np.size(self.th))
#     dth[0] = self.th[1] - self.th[0]
#     dth[1:-1] = ( self.th[2:] - self.th[1:-1] ) / 2 + ( self.th[1:-1] - self.th[0:-2] ) / 2
#     dth[-1] = self.th[-1] - self.th[-2]
#     return dth

# def regrid(f, th, S,f_out,th_out):
#     import numpy as np
#     from scipy.interpolate import interp2d
#     f_in = f.flatten()
#     th_in = th.flatten()
#     S_in = S

#     th_in=np.hstack((th_in-360,th_in,th_in+360)).flatten()
#     S_in=np.transpose(np.hstack((S_in,S_in,S_in)))

#     interpolator = interp2d(f_in,th_in,S_in)
#     S_out = np.transpose(interpolator(f_out,th_out))
#     return f_out, th_out, S_out
   


# In[33]:


#     def autoCorrect(self):
#         '''
#         autoCorrect()
        
#         Function to make all sorts of corrections for spectrum definitions
#             INPUTS:
#                 None
#             OUTPUTS: Update waveSpec instance
#                 None
                
#         TODO: Automatically reorder directions and spectra to be monotonic in direction
#         '''
#         import numpy as np

#         if np.size(self.f) == 0:
#             print("waveSpec:autoCorrect(): frequency array is empty")
#         if np.size(self.f) != 0 and np.size(self.f) != np.size(self.df):
#             #print("waveSpec:autoCorrect(): fixing df with f")
#             self.df = self._make_df()
#         if np.size(self.th) == 0:
#             print("waveSpec:autoCorrect(): direction array is empty")
#         if np.size(self.th) != 0 and np.size(self.th) != np.size(self.dth):
#             #print("waveSpec:autoCorrect(): fixing dth with th")
#             self.dth = self._make_dth()
#         if np.size(self.S) == 0:
#             print("waveSpec:autoCorrect(): full spectrum empty")
#         if np.size(self.S) != 0:
#             self.specType = "carpet"
#             if np.shape(self.S)[0] != np.shape(self.f)[0]:
#                 print("waveSpec:autoCorrect(): full spectrum S should have same size a f")
#             thOk = True
#             for i,S in enumerate(self.S):
#                 if len(np.shape(S)) == 0:
#                     print(f"waveSpec:autoCorrect(): full spectrum S[{i}] empty")
#                     thOk = False
#                     break
#                 if np.shape(S)[0] != np.shape(self.th)[0]:
#                     print(f"waveSpec:autoCorrect(): full spectrum S[{i}] should have same size a th")
#                     # cannot fix other stuff
#                     thOk = False
#                     break
#             if thOk:
#                 #print("waveSpec:autoCorrect(): recalculating Sf with S")
#                 # integrate full spectrum to get frequency spectrum
#                 nFre = np.shape(self.S)[0]
#                 self.Sf = np.zeros(nFre)
#                 for i,S in enumerate(self.S):
#                     self.Sf[i] = np.sum(S*self.dth)
                    
#                 #print("waveSpec:autoCorrect(): recalculating Sth with S")
#                 # integrate full spectrum to get directional spectrum
#                 nDir = np.shape(self.S)[1]
#                 self.Sth = np.zeros(nDir)
#                 St=np.transpose(self.S)
#                 for i,S in enumerate(St):
#                     #print(S)
#                     #print(self.df)
#                     self.Sth[i] = np.sum(S*self.df) 
#         else:
#             self.specType = "frequency"


# ## calc_dth

# In[ ]:


# def calc_dth(th):
#     """Calculate direction spacings
    
#     Args:
#         - th (nparray): Array of directions in deg 
        
#     Returns:
#         - dth (nparray): direction spacing for each direction

#     """
#     import numpy as np
#     dth = np.zeros(np.size(th))
#     dth[0] = th[1] - th[0]
#     dth[1:-1] = ( th[2:] - th[1:-1] ) / 2 + ( th[1:-1] - th[0:-2] ) / 2
#     dth[-1] = th[-1] - th[-2]
#     return dth  


# ## calc_Tp

# In[ ]:


# def calc_Tp(f, th, Sf):
#     """Peak period from the 2D spectrum
    
#     Args:
#         - f (nparray): Array of frequencies in Hz of input spectrum
#         - th (nparray): Array of directions in deg of input spectrum
#         - Sf (nparray): Array of frequency spectral densities in m^/(Hz) 
        
#     Returns:
#         - Tp (float): peak period of the wave spectrum [s]

#     """
#     import numpy as np
#     iTp = np.argmax(Sf)
#     return 1/self.f[iTp]


# ## calc_ThetaM

# In[ ]:


# def calc_ThetaM(f, th, S):
#     """Mean direction from the spectrum
    
#     Args:
#         - f (nparray): Array of frequencies in Hz of input spectrum
#         - th (nparray): Array of directions in deg of input spectrum
#         - S (nparray): Array of spectral densities in m^/(Hz.deg) of input spectrum
        
#     Returns:
#         - ThetaM (float): mean direction of the wave spectrum (same datum as spectrum directions)

#     """
#     import numpy as np
#     import numpy.matlib as ml
    
#     fm = np.transpose(ml.repmat(f,np.size(dth),1))
#     thm = ml.repmat(np.transpose(th),np.size(df),1) 
#     dfm = np.transpose(ml.repmat(df,np.size(dth),1))
#     dthm = ml.repmat(np.transpose(dth),np.size(df),1)
#     sth = np.sin(thm * np.pi/180)
#     cth = np.cos(thm * np.pi/180)
#     U = np.sum( np.sum( S * sth * dthm * dfm ))
#     L = np.sum( np.sum( S * cth * dthm * dfm ))
    
#     return ( (np.arctan2(U,L) * 180/np.pi) + 360 )%360


# ## EwansDoubleWrappedNormalSpreading

# In[ ]:


# def EwansDoubleWrappedNormalSpreading(d,dm1,dm2,s,n):
#     """Ewans Double wrapped normal wave spreading distribution
#     Args:
#         - d (1darray): directions [deg]
#         - dm1 (float): mean direction of peak 1 [deg]
#         - dm2 (float): mean direction of peak 2 [deg]
#         - s (float): angular width
#         - n (float): summation limit
#     Returns;
#         - y (ndarray): double wrapped normal distribution [1/rad]
#     Reference: 
#         - Ewans ...
    
#     TODO: Add reference
#     """
    
#     import numpy as np
#     d = d * np.pi/180
#     dm1 = dm1 * np.pi/180
#     dm2 = dm2 * np.pi/180
#     ld = len(d)
#     ldm1 = len(dm1)
#     s = s * np.pi/180
#     y = np.zeros([ldm1,ld])
#     for i in range(0,ldm1):
#         for k in np.arange(-n,n,1):
#             y[i,:] = y[i,:] + np.exp(-0.5*((d-dm1[i]-2*np.pi*k)/s[i])**2 ) + np.exp(-0.5*((d-dm2[i]-2*np.pi*k)/s[i])**2)
#         y[i,:] = y[i,:]/(np.sqrt(8*np.pi)*s[i])
#     # Make y a density
#     deld = np.diff(d)
#     deld = np.append(deld[1],deld)
#     sumYdelX = np.sum(y*deld)
#     y = y/sumYdelX
#     return y


# In[ ]:





# ## EwansDoubleWrappedNormalParameters

# In[ ]:


# def EwansDoubleWrappedNormalParameters(f,fp):
#     """Ewans (Maui) Bimodal wrapped normal wave spreading distribution parameters
#     Args:
#         - f (1darray): frequencies in Hz
#         - fp (float): peak frequency in Hz
#     Returns;
#         - y (ndarray): m by 3 matrix, such that
#                 y(:,1) is the angle (deg) of separation of peaks,
#                 y(:,2) is the amplitude [always one],
#                 y(:,3) is the std dev. (deg)
#     Reference: 
#         - Ewans ...
    
#     TODO: Add reference
#     """

#     import numpy as np
    
#     # A. Initial stuff
#     l = len(f)
#     y = np.nan*np.zeros([l,3])
#     y[:,1] = np.ones([l])
#     f = f/fp
#     i = np.where(f<1)
#     j = np.where(f>=1)

#     # B. Angular difference lf - (Ewans, Eqn 6.4)
#     y[i,0] = np.ones([len(i),1])*14.93

#     # C. Angular difference hf - (Ewans, Eqn 6.4)
#     a = 5.453
#     b = -2.750
#     y[j,0] = np.exp(a+b/(f[j]))

#     # D. Std dev. lf - (Ewans, Eqn 6.5)
#     a = 11.38
#     b = 5.357
#     c = -7.929
#     y[i,2] = a+b*(f[i])**c

#     # E. Std dev. hf - (Ewans, Eqn 6.5)
#     a = 32.13
#     b = -15.39
#     y[j,2] = a+b/(f[j]*f[j])

#     # F. Restrict Std dev. to be < 90;
#     k = np.where(y[:,2]>90)
#     y[k,2] = 90

#     return y    


# ## specIntParm

# In[ ]:


# def specIntParm(f, th, S):
#     import numpy as np
#     import numpy.matlib as ml

#     # 0 Matrix versions
#     fm = np.transpose(ml.repmat(f,np.size(dth),1))
#     thm = ml.repmat(np.transpose(th),np.size(df),1) 
#     dfm = np.transpose(ml.repmat(df,np.size(dth),1))
#     dthm = ml.repmat(np.transpose(dth),np.size(df),1)

#     # A. Tail moments
#     fcut = f[-1] + df[-1]/2
#     ecut = S[-1]
#     m0t = 1/4 * fcut    * ecut
#     m1t = 1/3 * fcut**2 * ecut
#     m2t = 1/2 * fcut**3 * ecut
#     m0t = 0
#     m1t = 0
#     m2t = 0

#     # B. Spectral moments
#     m0  = np.sum ( np.sum(           dthm * dfm * S    ,0) + m0t )
#     m1  = np.sum ( np.sum(fm       * dthm * dfm * S    ,0) + m1t )
#     m2  = np.sum ( np.sum(fm**2    * dthm * dfm * S    ,0) + m2t )
#     mm1 = np.sum ( np.sum(fm**(-1) * dthm * dfm * S    ,0)       )
#     fe4 = np.sum ( np.sum(fm       * dthm * dfm * S**4 ,0)       )
#     e4  = np.sum ( np.sum(           dthm * dfm * S**4 ,0)       )

#     # C. Sig Wave Height
#     Hm0 = 4 * np.sqrt(m0)

#     # D. Spectral Peak wave period
#     iTp = np.argmax(Sf)
#     Tp = 1/f[iTp]

#     # E. Mean Periods
#     T01 = m0/m1
#     T02 = np.sqrt(m0/m2)
#     Tm01 = mm1 / m0 

#     # F. T4 Young Ocean Eng. Vol 22 No.7 pp 669 to 686
#     T4 = 1/(fe4/e4)

#     # G. Directional parameters
#     iThetaP = np.argmax(Sth)
#     ThetaP = th[iThetaP]

#     sth = np.sin(thm * np.pi/180)
#     cth = np.cos(thm * np.pi/180)
#     U = np.sum( np.sum( S * sth * dthm * dfm ))
#     L = np.sum( np.sum( S * cth * dthm * dfm ))
#     ThetaM = ( (np.arctan2(U,L) * 180/np.pi) + 360 )%360

#     return [Hm0,Tp,T01,T02,Tm01,T4,ThetaP,ThetaM]

