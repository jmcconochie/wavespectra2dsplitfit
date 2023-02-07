#!/usr/bin/env python
# coding: utf-8

# # Simple Test for the wavespectra2dsplitfit package

# In[1]:


import numpy as np
from wavespectra2dsplitfit.S2DFit import readWaveSpectrum_mat
filename = 'data/ExampleWaveSpectraObservations.mat'
f, th, S, sDate = readWaveSpectrum_mat(filename)
S = S * np.pi/180 # convert from m^2/(Hz.rad) to m^2/(Hz.deg)
   
# Setup fitting configuration - simple example with no wind (also usually best setup with no wind)    
tConfig = {
    'maxPartitions': 3,
    'useClustering': True,
    'useWind': False,
    'useFittedWindSea': False, 
    'useWindSeaInClustering': False,
}

# Just do the first spectrum
from wavespectra2dsplitfit.S2DFit import fit2DSpectrum
specParms, fitStatus, diagOut = fit2DSpectrum(f[0], th[0], S[0,:,:], **tConfig)
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

from wavespectra2dsplitfit.S2DFit import plot2DFittingDiagnostics
f, th, S, f_sm, th_sm, S_sm, wsMask, Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus, partitionMap, S_t, Sparts_t, Hs_parts_input, Hs_parts_recon = diagOut
plot2DFittingDiagnostics(
    specParms, 
    f, th, S, 
    f_sm, th_sm, S_sm, 
    wsMask,
    Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus,
    tConfig['useWind'], tConfig['useClustering'],
    saveFigFilename = 'test',  
    tag = "S2DFit Simple Test"  
)


# In[2]:


try:
    get_ipython() 
    get_ipython().system('jupyter nbconvert test.ipynb --to python')
except:
    None

