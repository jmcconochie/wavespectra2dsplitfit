#!/usr/bin/env python
# coding: utf-8

# # Simple Test for the wavespectra2dsplitfit package

# In[2]:


# A. Read spectra data from a matlab file
def readspec_mat(filename, dates="td", freq="fd", dirn="thetad", spec2d="spec2d"):  
    # Read wave spectra from a matlab file
    # variables should be:
    #   td[nTimes] - vector of matlab date serials
    #   fd[nFre] - vector of wave frequencies in Hz
    #   thetad[nDir] - vector of wave directions in degrees
    #   spec2d[nTimes,nFre,nDir] - array of 2D wave spectra for each time in m^2/(Hz.deg)
    import numpy as np
    import scipy.io
    mat = scipy.io.loadmat(filename)
    mat.keys()
    tm = mat[dates]
    f = mat[freq]
    th = mat[dirn]
    S = mat[spec2d] * np.pi/180

    
    import datetime as dt
    sDate = [dt.datetime(x[0],x[1],x[2],x[3],x[4],x[5]) for x in tm]

    from wavespectra2dsplitfit.wavespec import waveSpec
    allSpec = [waveSpec() for x in sDate]
    for i,tSpec in enumerate(allSpec):
        tSpec.f = f[0]
        tSpec.th = th[0]
        tSpec.S = S[i,:,:]
        tSpec.autoCorrect()
        tSpec.meta = {'date':sDate[i]}
    
    return allSpec

filename = 'data/ExampleWaveSpectraObservations.mat'
rawSpec = readspec_mat(filename)
   
# Setup fitting configuration - simple example with no wind (also usually best setup with no wind)    
tConfig = {
    'maxPartitions': 3,
    'useClustering': True,
    'useWind': False,
    'useFittedWindSea': False, 
    'useWindSeaInClustering': False,
    'doPlot': True,
    'saveFigFilename': "test.png"
}       

# Just do the first spectrum
specParms, fitStatus = rawSpec[0].fit2DSpectrum(tConfig)
print(specParms, fitStatus)

for tSpec in specParms:
    print("===== PARTITION =====")
    print("Hs = ",tSpec[0])
    print("Tp = ",tSpec[1])
    print("Gamma = ",tSpec[2])
    print("Sigma A = ",tSpec[3])
    print("Sigma B = ",tSpec[4])
    print("Tail Exp = ",tSpec[5])
    print("ThetaP = ",tSpec[6])
print("===== FITTING OUTCOME =====")
print(f"Fitting successful: ",fitStatus[0])
print(f"RMS error of fit: ",fitStatus[1])
print(f"Number of function evalutions: ",fitStatus[2])


# In[3]:


try:
    get_ipython() 
    get_ipython().system('jupyter nbconvert test.ipynb --to python')
except:
    None


# In[ ]:




