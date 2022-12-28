#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    None


# # FUNCTION: runFitting

# In[2]:


def runFitting(tag, f, th, S, sDate, timeRange, baseConfig):
    import numpy as np
    
    # B. Loop over times and fit spectra   
    allPartRes = []
    allPartStatus = []
    for iTime in timeRange:
        fTime = sDate[iTime].strftime("%Y-%m-%dT%H_%M_%S")
        
        # B1. Do the fitting
        print("=== START =====================================================================")
        print(iTime,fTime,baseConfig)
        from wavespectra2dsplitfit.S2DFit import fit2DSpectrum
        specParms, fitStatus, diagOut = fit2DSpectrum(f, th, S[iTime,:,:], **baseConfig)
        print(specParms, fitStatus)
        print("=== END =======================================================================")

        from wavespectra2dsplitfit.S2DFit import plot2DFittingDiagnostics
        df, dth, dS, f_sm, th_sm, S_sm, wsMask, Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus = diagOut
        plot2DFittingDiagnostics(
            specParms, 
            df, dth, dS, 
            f_sm, th_sm, S_sm, 
            wsMask,
            Tp_pk, ThetaP_pk, Tp_sel, ThetaP_sel, whichClus,
            baseConfig['useWind'], baseConfig['useClustering'],
            saveFigFilename = f"images/{tag}_{fTime}",  
            tag = tag
        )
        
        # B2. Save off all the parameters
        allPartRes.append(specParms)
        allPartStatus.append(fitStatus)                
    
    return allPartRes, allPartStatus


# # FUNCTION: fitResults2Pandas

# In[3]:


def fitResults2Pandas(allPartRes, allPartStatus, allTimes, timeRange):

    # C. Save all the parameters to a csv file
    import pandas as pd
    # C1. Get maximum elements for all partitions
    maxParts = 0
    for tParts in allPartRes:
        tRes = [x for xs in tParts for x in xs] 
        maxParts = max(len(tRes),maxParts)

    # C2. Load data table with partition results and fitting results
    import numpy as np
    nTimesUsed = len(allPartRes)
    dataArray = np.zeros([nTimesUsed, maxParts + 3])
    selTimes = []
    for iTime,tParts in enumerate(allPartRes):
        tRes = [x for xs in allPartRes[iTime] for x in xs] 
        dataArray[iTime,0:len(tRes)] = tRes
        dataArray[iTime,-3:] = allPartStatus[iTime]
        selTimes.append(allTimes[timeRange[iTime]])

    columns = int(maxParts/8) * ["Hs","Tp","gamma","sigmaa","sigmab","tailExp","ThetaP","sSpread"]
    columns = columns + ["Solved","Fit Error","NoFuncEvals"]
    df = pd.DataFrame(index = selTimes, data = dataArray, columns = columns)
    return df


# # Main 

# In[6]:


def main(filename):
    
    # A. Read the data file 
    import numpy as np
    from wavespectra2dsplitfit.S2DFit import readWaveSpectrum_mat
    f, th, S, sDate = readWaveSpectrum_mat(filename)
    S = S * np.pi/180 # convert from m^2/(Hz.rad) to m^2/(Hz.deg)

    # B. Regrid spectra to increase speed of fitting
    import numpy as np
    from wavespectra2dsplitfit.S2DFit import interpSpectrum
    f_out = np.arange(0.04,0.4,0.01)
    th_out = np.arange(0,350,15)
    S_out = np.zeros((len(sDate),len(f_out),len(th_out)))
    for i in range(0,len(sDate),1):
        S_out[i,:,:] = interpSpectrum(f, th, S[i,:,:], f_out, th_out)

    # C. Run the fitting
    baseConfig = {
        'maxPartitions': 3,
        'useClustering': True,
        'useWind': False,
        'useFittedWindSea': False, 
        'useWindSeaInClustering': False,
    }
    
    timeRange = range(0,len(sDate),1)
    #timeRange = range(0,2,1)
    allPartRes, allPartStatus = runFitting("Test", f_out, th_out, S_out, sDate, timeRange, baseConfig)   

    # D. Compile and save the results
    df = fitResults2Pandas(allPartRes, allPartStatus, sDate, timeRange) 
    df.to_csv(f"fittedParms.csv")

    return df


# In[7]:


try:
    # If running in jupyter notebook put filename here
    get_ipython()  # fails if not running in jupyter notebook/ipython 
    filename = "data/ExampleWaveSpectraObservations.mat"
except:
    # If runing from command line specify input matlab file as command line
    #  argument e.g. python fitFromMatlabFile.py data/ExampleWaveSpectraObservations.mat
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
df = main(filename)


# In[ ]:


try:
    get_ipython() 
    get_ipython().system('jupyter nbconvert fitFromMatlabFile.ipynb --to python')
except:
    None

