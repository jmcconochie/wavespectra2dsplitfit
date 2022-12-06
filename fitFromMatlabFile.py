#!/usr/bin/env python
# coding: utf-8

# # FUNCTION: runFitting

# In[14]:


def runFitting(tag, rawSpec, allTimes, timeRange, baseConfig):
    import numpy as np
    
    # B. Loop over times and fit spectra   
    allPartRes = []
    allPartStatus = []
    for iTime in timeRange:
        fTime = rawSpec[iTime].meta["date"].strftime("%Y-%m-%dT%H_%M_%S")
        
        # B3. Make the full configuration
        aConfig = {
            'iTime': iTime,
            'fTime': fTime,
            'saveFigFilename': f"images/{tag}_{fTime}"
        }
        tConfig = {**baseConfig, **aConfig}
        if tConfig['useWind']:
            tConfig['wspd'] = rawSpec[iTime].meta['wspd']
            tConfig['wdir'] = rawSpec[iTime].meta['wdir']
            tConfig['dpt'] = rawSpec[iTime].meta['dpt']
        else:
            tConfig['wspd'] = None
            tConfig['wdir'] = None
            tConfig['dpt'] = None
        
        # B4. Do the fitting
        print("=== START =====================================================================")
        print(iTime,fTime,tConfig)
        specParms, fitStatus = rawSpec[iTime].fit2DSpectrum(tConfig)
        print(specParms, fitStatus)
        print("=== END =======================================================================")
        
        # B5. Save off all the parameters
        allPartRes.append(specParms)
        allPartStatus.append(fitStatus)                
    
    return allPartRes, allPartStatus


# # FUNCTION: fitResults2Pandas

# In[15]:


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


# # FUNCTION: readspec_mat

# In[16]:


def readspec_mat(filename, dates="td", freq="fd", dirn="thetad", spec2d="spec2d"):  

    import scipy.io
    mat = scipy.io.loadmat(filename)
    mat.keys()
    tm = mat[dates]
    f = mat[freq]
    th = mat[dirn]
    S = mat[spec2d]

    import numpy as np
    import datetime as dt
    sDate = [dt.datetime(x[0],x[1],x[2],x[3],x[4],x[5]) for x in tm]

    from wavespectra2dsplitfit.wavespec import waveSpec
    #from wavespectra2dsplitfit.wavespectra2dsplitfit.wavespec import waveSpec
    allSpec = [waveSpec() for x in sDate]
    for i,tSpec in enumerate(allSpec):
        tSpec.f = f[0]
        tSpec.th = th[0]
        tSpec.S = S[i,:,:]
        tSpec.autoCorrect()
        tSpec.meta = {'date':sDate[i]}
    
    return allSpec


# # Main 

# In[17]:


def main(filename):
    
    # A. Read the data file 
    #filename = 'data/Prelude_RPS_realtime_201901.mat'
    allSpec = readspec_mat(filename)

    # B. Regrid spectra to increase speed of fitting
    import numpy as np
    f = np.arange(0.04,0.4,0.01)
    th = np.arange(0,350,15)
    for tSpec in allSpec:
        tSpec.regrid(f,th) 

    # C. Run the fitting
    baseConfig = {
        'maxPartitions': 3,
        'useClustering': True,
        'useWind': False,
        'useFittedWindSea': False, 
        'useWindSeaInClustering': False,
        'plotClusterSpace': False,
        'doPlot': True
    }
    allTimes = [x.meta['date'] for x in allSpec]   
    timeRange = range(0,len(allTimes),1)
    #timeRange = range(0,10,1)
    allPartRes, allPartStatus = runFitting("Test", allSpec, allTimes, timeRange, baseConfig)   

    # D. Compile and save the results
    df = fitResults2Pandas(allPartRes, allPartStatus, allTimes, timeRange) 
    df.to_csv(f"fittedParms.csv")

    return df


# In[ ]:


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


# In[ ]:




