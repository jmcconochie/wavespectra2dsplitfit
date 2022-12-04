# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 21:45:58 2022

TODO: Rename functions and write descriptions and give examples for all
TODO: Sort out logic for all options

@author: J.McConochie
"""

# Make JONSWAP
class waveSpec:
    '''
    waveSpec Class

    Functions to carry out spectral fitting for wave spectrum using 
        JONSWAP and cos2s.  Will fit nPartitions based on a peak selection.
        
    
    Wave spectrum data can be loaded into class properties:
        f[nFre] - frequency [Hz]
        df[nFre] - frequency steps [Hz]
        th[nDir] - directions [deg]
        dth[nDir] - direction steps [deg], relative to N, coming from 
        S[nFre,nDir] - carpet spectrum [m^2/(Hz.deg)]
        Sf[nFre] - frequency spectrum [m^2/Hz]
        Sth[nDir] - direction spectrum [m^2/deg]
    
    Main functions to use are:
        autoCorrect
            will check the loaded data into the properties and
            fill in missing data from available data (e.g. calculates df from f)
        fitMulitDirJONSWAPCos2s
            Carry out n x JONSWAP & cos2s spectrum fitting with fixed Tp and ThetaP inputs
            Solves for Hs, gamma, s, and tail exponent (maybe fixed)
            Conserve HsSwell, HsSea, HsTotal (after partitioning)
    '''
    
    def __init__(self):
        import numpy as np
        
        self.f = np.arange(0.01,0.51,0.01) # Hz
        self.df = self._make_df() # Hz
        self.th = np.arange(0,360,5) # deg
        self.dth = self._make_dth() # deg
      
        self.Sf = np.array([]) # m^2 / Hz
        self.Sth = np.array([]) # m^2 / (deg)
        
        self.S = np.array(np.zeros([len(self.f),len(self.th)])) # m^2 / (Hz.deg) size = [nf][nth]
        self.specType = ""
        
        self.meta = {}
        self.autoCorrect()

    def __repr__(self):
        import numpy as np
        o = "waveSpec Shapes\n"
        o += f"f {np.shape(self.f)}\n" 
        o += f"th {np.shape(self.th)}\n" 
        o += f"S {np.shape(self.S)}\n" 
        o += "Frequencies:\n"
        o += f"{self.f}\n"
        o += "Directions:\n"
        o += f"{self.th}\n" 
        o += "Spectra:\n"
        o += f"{self.S}\n" 
        return o
    
    def __str__(self):
        return __repr__(self)
        

    def _make_df(self):
        import numpy as np
        df = np.zeros(np.size(self.f))
        df[0] = self.f[1] - self.f[0]
        df[1:-1] = ( self.f[2:] - self.f[1:-1] ) / 2 + ( self.f[1:-1] - self.f[0:-2] ) / 2
        df[-1] = self.f[-1] - self.f[-2]
        return df
    
    
    def _make_dth(self):
        import numpy as np

        dth = np.zeros(np.size(self.th))
        dth[0] = self.th[1] - self.th[0]
        dth[1:-1] = ( self.th[2:] - self.th[1:-1] ) / 2 + ( self.th[1:-1] - self.th[0:-2] ) / 2
        dth[-1] = self.th[-1] - self.th[-2]
        return dth
    
        
    def makeJONSWAP1D(self,fSpecParm):
        '''
        makeJONSWAP1D(fSpecParm)
        
        Function to make a JONSWAP spectrum
            INPUTS:
                fSpecParm - [Hs,Tp,gamma,sigmaa,sigmab,jexp]
                waveSpec instance must have f array defined; spectrum is created on these frequencies
            OUTPUTS: Update waveSpec instance
                None
        '''
        import numpy as np

        if np.size(self.f) == 0:
            print("waveSpec:autoCorrect(): frequency array is empty")
            return
        
        Hs,Tp,gamma,sigmaa,sigmab,jexp = fSpecParm
        
        g = 9.81 # [m/s^2]
        fp = 1 / Tp
        sigma = sigmaa * (self.f < fp) + sigmab * (self.f >= fp) 
        G = gamma ** np.exp( -1 * ( ((self.f-fp) ** 2) / ( 2 * sigma**2 * fp**2)) )
        alpha = 1 # will be rescaled
        Sf = alpha * g**2 * (2.*np.pi)**(jexp+1) * self.f**jexp * np.exp( jexp/4 * (self.f/fp)**-4 ) * G # m^2/Hz
        calcHs = 4 * np.sqrt(np.trapz(Sf,self.f))
        if calcHs == 0.0:
            self.Sf = Sf 
            self.Sth = np.array([])
            self.S = np.array([])
        else:
            self.Sf = Sf * (Hs / calcHs)**2
            self.Sth = np.array([])
            self.S = np.array([])
        
        
    def applyCos2s(self,dSpecParm,spreadType='parametric',TpSeaSwellCut=9):
        '''
        applyCos2s(dSpecParm)
        
        Function to apply cos2s spreading to waveSpec instance using th and Sf
            INPUTS:
                dSpecParm - [waveDir,s]
                    waveDir - main wave direction in [deg]
                    s - spread 
                    spreadType  if 'parametric' s parameter is ignore and will use 
                        Ewans (1998) for wind Tp < TpSeaSwellCut and Ewans (2001) for Tp > TpSeaSwellCut
                    provide Tp instead of s in dSpecParm
            OUTPUTS: Update waveSpec instance
                S is created with th, Sf with cos2s in direction waveDir with spreading s
                Sth is updating using autoCorrect() method
        '''
        import numpy as np
        import numpy.matlib as ml
        
        useEwansCos2s = True
        if (spreadType == 'parametric'):
            waveDir,Tp = dSpecParm
            if Tp < TpSeaSwellCut:
                if useEwansCos2s:
                    # Use Ewans 1998
                    s = np.zeros(len(self.f))
                    ffp = self.f * Tp
                    for iFreq,tFreq in enumerate(self.f):
                        if ffp[iFreq] < 1:
                            s[iFreq] = 15.5*ffp[iFreq]**9.47
                        else:
                            s[iFreq] = 13.1*ffp[iFreq]**-1.94
                    Dth = self.cos2s(waveDir,s)
                    #import matplotlib.pyplot as plt
                    #plt.figure()
                    #plt.pcolormesh(self.f,self.th,np.transpose(Dth))
                    #plt.colorbar()
                    self.S = np.zeros((len(self.f),len(self.th)))
                    for i,f in enumerate(self.f):
                        self.S[i,:] = self.Sf[i] * Dth[i,:] * np.pi/180
                    # plt.figure()
                    # plt.pcolormesh(self.f,self.th,np.transpose(self.S)))
                else:
                    # Use Ewans with bifurcation
                    o = self.MauiBimodalWN(self.f,1/Tp)
                    dm1 = (waveDir-o[:,0]/2) % 360
                    dm2 = (waveDir+o[:,0]/2) % 360
                    m1 = dm1 < 360;  dm1[m1] = dm1[m1] + 360
                    m2 = dm2 > 360;  dm2[m2] = dm2[m2] - 360
                    s = o[:,2]
                    n = 5
                    
                    Dth = self.dblwn(self.th,dm1,dm2,s,n)
                    #import matplotlib.pyplot as plt
                    #plt.figure()
                    #plt.pcolormesh(self.f,self.th,np.transpose(Dth))
                    #for i,f in enumerate(self.th):
                    #    print(np.trapz(self.th,Dth[:,i]))
                    #plt.colorbar()
                    
                    self.S = np.transpose(ml.repmat(self.Sf,len(self.th),1)) * Dth #* np.pi/180
            else:
                # User Ewans 2001 swell
                Dth = self.EwansWrappedNormal(waveDir,Tp)
                self.S = np.transpose(ml.repmat(self.Sf,len(self.th),1)) * Dth * np.pi/180
        else: 
            waveDir,s = dSpecParm
            Dth = self.cos2s(waveDir,s)
            self.S = np.zeros((len(self.f),len(self.th)))
            for i,f in enumerate(self.f):
                self.S[i] = self.Sf[i] * Dth * np.pi/180
                
        self.autoCorrect()

            
    def makeJONSWAP2D(self,fSpecParm,dSpecParm,spreadType='parametric'):
        '''
        makeJONSWAP2D(fSpecParm,dSpecParm)
        
        Function to make a JONSWAP spectrum
            INPUTS:
                fSpecParm - [Hs,Tp,gamma,sigmaa,sigmab,jexp]
                dSpecParm - [waveDir,s]
                spreadType - if 'parametric' s parameter is ignore and will use 
                    Ewans (1998) for wind Tp < 9s and Ewans (2001 Wrapped Normal) for Tp > 9s
                waveSpec instance must have f array defined; spectrum is created on these frequencies
            OUTPUTS: Update waveSpec instance
                None
        '''
        self.makeJONSWAP1D(fSpecParm)
        self.applyCos2s(dSpecParm,spreadType)
        

    def autoCorrect(self):
        '''
        autoCorrect()
        
        Function to make all sorts of corrections for spectrum definitions
            INPUTS:
                None
            OUTPUTS: Update waveSpec instance
                None
                
        TODO: Automatically reorder directions and spectra to be monotonic in direction
        '''
        import numpy as np

        if np.size(self.f) == 0:
            print("waveSpec:autoCorrect(): frequency array is empty")
        if np.size(self.f) != 0 and np.size(self.f) != np.size(self.df):
            #print("waveSpec:autoCorrect(): fixing df with f")
            self.df = self._make_df()
        if np.size(self.th) == 0:
            print("waveSpec:autoCorrect(): direction array is empty")
        if np.size(self.th) != 0 and np.size(self.th) != np.size(self.dth):
            #print("waveSpec:autoCorrect(): fixing dth with th")
            self.dth = self._make_dth()
        if np.size(self.S) == 0:
            print("waveSpec:autoCorrect(): full spectrum empty")
        if np.size(self.S) != 0:
            self.specType = "carpet"
            if np.shape(self.S)[0] != np.shape(self.f)[0]:
                print("waveSpec:autoCorrect(): full spectrum S should have same size a f")
            thOk = True
            for i,S in enumerate(self.S):
                if len(np.shape(S)) == 0:
                    print(f"waveSpec:autoCorrect(): full spectrum S[{i}] empty")
                    thOk = False
                    break
                if np.shape(S)[0] != np.shape(self.th)[0]:
                    print(f"waveSpec:autoCorrect(): full spectrum S[{i}] should have same size a th")
                    # cannot fix other stuff
                    thOk = False
                    break
            if thOk:
                #print("waveSpec:autoCorrect(): recalculating Sf with S")
                # integrate full spectrum to get frequency spectrum
                nFre = np.shape(self.S)[0]
                self.Sf = np.zeros(nFre)
                for i,S in enumerate(self.S):
                    self.Sf[i] = np.sum(S*self.dth)
                    
                #print("waveSpec:autoCorrect(): recalculating Sth with S")
                # integrate full spectrum to get directional spectrum
                nDir = np.shape(self.S)[1]
                self.Sth = np.zeros(nDir)
                St=np.transpose(self.S)
                for i,S in enumerate(St):
                    #print(S)
                    #print(self.df)
                    self.Sth[i] = np.sum(S*self.df) 
        else:
            self.specType = "frequency"
            
       
    
    def rmsSpread2s(self,rmsSpread):
        '''
        rmsSpread2s(rmsSpread)
        
        Function to convert wave spreading [deg] to cos2s s parameter
        INPUT:
            rmsSpread - RMS spread in degrees
        OUTPUT:
            s - equivalent s parameter for cos2s spreading function
        '''
        import numpy as np
        if rmsSpread < 0.1:
            rmsSpread = 0.1
        if rmsSpread > 60:
            rmsSpread = 60
        return 2 / (rmsSpread * np.pi/180.0)**2 - 1
    
    
    def angDiff(self,fromAngle,toAngle):
        '''
        cos2s(fromAngle,toAngle)
        
        Calculate the difference in angle fromAngle relative to toAngle
        INPUT:
            fromAngle - vector of from angles in [deg]
            toAngle - to angle in [deg]
        OUTPUT:
            angDiff - difference in angle fromAngle relative to toAngle
        '''
        import numpy as np

        ang1 = np.mod(fromAngle,360.)
        ang2 = np.mod(toAngle,360.)
        angDiff1 = ang1-ang2
        
        m = angDiff1 > 180.
        angDiff1[m] -= 360.
        m = angDiff1 < -180.
        angDiff1[m] += 360.
        return angDiff1
    
    
    def cos2s(self,waveDir,s):
        '''
        cos2s(waveDir,s)
        
        Function to create a 1D normalised direction function using cos2s
        INPUT:
            waveDir - main wave direction in [deg]
            s - spread 
        OUTPUT:
            Dth - vector of the normalised cos2s spread function for each direction
                in the waveSpec instance th [deg] array
                - in case of frequency dependent spreading this is size of 2D Spectrum
        '''
        import numpy as np
        import scipy.special as ss
        
        def dth(s):
            A = (2 ** (2*s-1)*ss.gamma(s+1)**2) / (np.pi * ss.gamma(2*s+1))
            dDir = np.abs(self.angDiff(self.th,waveDir))
            Dth = A * ( np.cos( 0.5 * dDir * np.pi/180) ** (2 * s) )
            return Dth
            
        if len(s) == len(self.f):
            # frequency dependent s
            Dth = np.zeros([len(self.f),len(self.th)]) 
            for iFreq, tFreq in enumerate(self.f):
                Dth[iFreq,:] = dth(s[iFreq]) 
            #import matplotlib.pyplot as plt                
            #plt.figure()
            #plt.pcolormesh(self.f,self.th,np.transpose(Dth))
            
        else:
            Dth = dth(s)

        return Dth

    def dblwn(self,d,dm1,dm2,s,n):
    # dblwn_SDA Double wrapped normal distribution
    #     function y = dblwn_SDA(d,dm1,dm2,s,n)
    #        where d is the vector of directions,
    #              dm1 is the mean direction of peak 1,
    #              dm2 is the mean direction of peak 2,
    #              s is the angular width,
    #              n is the summation limit, and
    #              y is the double wrapped normal distribution.
    #
    # KCE:10-Nov-22
    # Converted to python Jason McConochie, 11Nov2022
        import numpy as np
        d = d * np.pi/180
        dm1 = dm1 * np.pi/180
        dm2 = dm2 * np.pi/180
        ld = len(d)
        ldm1 = len(dm1)	# if ldm1>1 then assume there are several frequencies given
        s = s * np.pi/180
        y = np.zeros([ldm1,ld])
        for i in range(0,ldm1):
           for k in np.arange(-n,n,1):
              y[i,:] = y[i,:] + np.exp(-0.5*((d-dm1[i]-2*np.pi*k)/s[i])**2 ) + np.exp(-0.5*((d-dm2[i]-2*np.pi*k)/s[i])**2)
           y[i,:] = y[i,:]/(np.sqrt(8*np.pi)*s[i])
        
        # Make y a density
        #deld = np.diff(d*180/np.pi)
        deld = np.diff(d)
        deld = np.append(deld[1],deld)
        sumYdelX = np.sum(y*deld)
        y = y/sumYdelX
        return y
    
    def MauiBimodalWN(self,f,fp):
        # MauiBimodalWN_SDA Maui bimodal spreading function, based on symmetric wrapped
        # normal
        # Functional form is y = MauiBimodalWN_SDA(f,fp)
        # 	where f is an m vector of frequencies in Hz,
        #       fp is the peak frequency in Hz,
        # 	    y is a m by 3 matrix, such that
        # 		y(:,1) is the angle (deg) of separation of peaks,
        # 		y(:,2) is the amplitude [always one],
        # 		y(:,3) is the std dev. (deg), and
        # Ref:Ewans JPO, March 1998.
        # KCE:10-Nov-22
        # Converted to python Jason McConochie, 11Nov2022
        import numpy as np
        l = len(f)
        y = np.nan*np.zeros([l,3])
        y[:,1] = np.ones([l])
        f = f/fp
        i = np.where(f<1)
        j = np.where(f>=1)
        
        # **** Angular difference lf - (Ewans, Eqn 6.4)
        y[i,0] = np.ones([len(i),1])*14.93
        
        # **** Angular difference hf - (Ewans, Eqn 6.4)
        a = 5.453
        b = -2.750
        y[j,0] = np.exp(a+b/(f[j]))
        
        # **** Std dev. lf - (Ewans, Eqn 6.5)
        a = 11.38
        b = 5.357
        c = -7.929
        y[i,2] = a+b*(f[i])**c
        
        # **** Std dev. hf - (Ewans, Eqn 6.5)
        a = 32.13
        b = -15.39
        y[j,2] = a+b/(f[j]*f[j])
        
        # *** Restrict Std dev. to be < 90;
        k = np.where(y[:,2]>90)
        y[k,2] = 90
        
        return y


    def EwansWrappedNormal(self,waveDir,Tp):
    #wn_SDA Wrapped Normal spreading function.
    #	Functional form is y = wn_SDA(x,x0,std)
    #		where	x is the numpy vector of angles (in degrees),
    #			    x0 is the mean direction (in degrees), and
    
    #               y[nth] is the spreading density in rad^(-1);
    # Example:
    #   x = np.arange(0,360,5)
    #   x0 = 270
    #   s = 15
    # KCE: 9-Nov-22
    # JdM adapated to python: 11 Nov 2022
    
    # x = np.arange(0,360,5)
    # x0 = 270
    # s = 15
        x0 = waveDir
        x = self.th
        import numpy as np
        fp = 1 / Tp
        # sigma_wn is the standard deviation
        yo = np.zeros([len(self.f),len(self.th)])
        for iFreq,f in enumerate(self.f):
            ffp = f / fp
            if ffp < 1.0:
                a = 6; b = 4; c = -5    
            else:
                a = -36; b = 46; c = 0.3
            sigma_wn = a + b * ffp ** c
            
            d = x * np.pi/180.
            d0 = x0 * np.pi/180.
            s = sigma_wn * np.pi/180.
            if s < 1:
            	y = np.zeros(len(x))
            	for i in range(-1,2):
            		y = y + np.exp(-1/2*((d-d0-2*np.pi*i)/s)**2)/(np.sqrt(2*np.pi)*s)
            else:
            	y = np.ones(len(x))*1/(2*np.pi)
            	for i in range(1,6):	
            		y = y + np.exp(-i*i*s*s/2)/np.pi*np.cos(i*d-i*d0)
            # *** make y a density
            delx = np.diff(x) * np.pi/180
            delx = np.append(delx[1],delx)
            sumYdelX = np.sum(y*delx)
            yo[iFreq,:] = y/sumYdelX   # need to add pi/180 to normalise
            #import matplotlib.pyplot as plt            
            #plt.plot(x,y)     
            #np.trapz(y*np.pi/180,x)
            #plt.pcolormesh(self.f,self.th,np.transpose(d))
        return yo




    
    def specIntParm(self):
        import numpy as np
        import numpy.matlib as ml

        # 0 Matrix versions
        fm = np.transpose(ml.repmat(self.f,np.size(self.dth),1))
        thm = ml.repmat(np.transpose(self.th),np.size(self.df),1) 
        dfm = np.transpose(ml.repmat(self.df,np.size(self.dth),1))
        dthm = ml.repmat(np.transpose(self.dth),np.size(self.df),1)

        # A. Tail moments
        fcut = self.f[-1] + self.df[-1]/2
        ecut = self.S[-1]
        m0t = 1/4 * fcut    * ecut
        m1t = 1/3 * fcut**2 * ecut
        m2t = 1/2 * fcut**3 * ecut
        m0t = 0
        m1t = 0
        m2t = 0

        # B. Spectral moments
        m0  = np.sum ( np.sum(           dthm * dfm * self.S    ,0) + m0t )
        m1  = np.sum ( np.sum(fm       * dthm * dfm * self.S    ,0) + m1t )
        m2  = np.sum ( np.sum(fm**2    * dthm * dfm * self.S    ,0) + m2t )
        mm1 = np.sum ( np.sum(fm**(-1) * dthm * dfm * self.S    ,0)       )
        fe4 = np.sum ( np.sum(fm       * dthm * dfm * self.S**4 ,0)       )
        e4  = np.sum ( np.sum(           dthm * dfm * self.S**4 ,0)       )

        # C. Sig Wave Height
        Hm0 = 4 * np.sqrt(m0)

        # D. Spectral Peak wave period
        iTp = np.argmax(self.Sf)
        Tp = 1/self.f[iTp]

        # E. Mean Periods
        T01 = m0/m1
        T02 = np.sqrt(m0/m2)
        Tm01 = mm1 / m0 

        # F. T4 Young Ocean Eng. Vol 22 No.7 pp 669 to 686
        T4 = 1/(fe4/e4)

        # G. Directional parameters
        iThetaP = np.argmax(self.Sth)
        ThetaP = self.th[iThetaP]

        sth = np.sin(thm * np.pi/180)
        cth = np.cos(thm * np.pi/180)
        U = np.sum( np.sum( self.S * sth * dthm * dfm ))
        L = np.sum( np.sum( self.S * cth * dthm * dfm ))
        ThetaM = ( (np.arctan2(U,L) * 180/np.pi) + 360 )%360

        return [Hm0,Tp,T01,T02,Tm01,T4,ThetaP,ThetaM]

    

    def fitMulitDirJONSWAPCos2s(self,parmActive,parmStart,plot=False,spreadType='parametric'):
        '''
        fitMulitDirJONSWAPCos2s(parmActive,parmStart,plot=False)

        Function to carry out n x JONSWAP/cos2s spectral fitting for carpet spectrum
            INPUTS:
                parmActive - [[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread],....]
                    provide array of fixed parameters or True (if parameter to be fitted), for each partition
                parmStart - [[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread],....] 
                    procide starting parameters for each partition
            OUTPUTS: [parmFit1, ...] 
                JONSWAP and cos2s wave parameters [[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread],...] for each partition
        '''
        def specErrF(x,*args):
            import numpy as np

            inSpec = args[0]   # this is the input 2D spectrum
            parmActive = args[1] # [[Hs,Tp,gamma,sigmaa,sigmab,jexp,waveDir,s],..]
            plot = args[2]

            # Map only active arguments: parmActive - True or FixedValue
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
                
            # Make a new spectra
            sTot = waveSpec()
            sTot.f = np.array(inSpec.f)
            sTot.th = np.array(inSpec.th)
            sTot.S = inSpec.S * 0
            
            for iPart,vPart in enumerate(allPartParms):
                s = waveSpec()
                s.f = np.array(inSpec.f)
                s.th = np.array(inSpec.th)
                if spreadType == "parametric":
                    s.makeJONSWAP2D(vPart[0:6],np.array(vPart)[[6,1]],'parametric')   
                else:
                    s.makeJONSWAP2D(vPart[0:6],vPart[6:9],spreadType) 
                
                sTot.S = sTot.S + s.S
                #print("allPartParms:",iPart,vPart[0:6],vPart[6:9])
                #print(s.S)
    
            # Calculate error
            # def rmse(predictions, targets):
            #     mP = predictions > 1e-4
            #     mT = targets > 1e-4
            #     P = np.log(predictions[np.logical_and(mP,mT)])
            #     T = np.log(targets[np.logical_and(mP,mT)])
            #     return np.sqrt(np.sum(np.power(P - T,2)))
            
            # Calculate error
            def rmse(predictions, targets):
                return np.sqrt(np.sum(np.power(predictions - targets,2)))
            
            tErr=rmse(sTot.S,inSpec.S)
            if plot:
                if np.any(np.isnan(sTot.S)):
                    None
                    #print(sTot.S)
                    #print("allPartParms:",allPartParms)
                    #print("tErr:", tErr)
                    
            for iPart,vPart in enumerate(allPartParms):
                q = [
                    vPart[0]<0,  # Hs should not be below 0
                    vPart[2]<1,  # gamma should not be less than 1
                    vPart[2]>20,  # gamma should not be less than 1
                    vPart[5]>-1, # tail exponent should not be higher than -1
                    vPart[5]<-50, # tail exponent should not be lower than -8
                    vPart[7]<1,   # spread, s should not be less than 1
                    vPart[7]>25   # spread, s should not be greater than 20
                    ]
                if np.any(q):
                     #print(allPartParms)
                     tErr = 1e5
            

            return tErr
        

        # Run the fitting routine
        
        # Map only active arguments: parmStart 
        allParmStart = []
        for iPart,vPart in enumerate(parmActive):
            for iParm,vParm in enumerate(vPart):
                if vParm == True:
                    allParmStart.append(parmStart[iPart][iParm])
            
        #print("start:",allParmStart)
        maxIter = 500 * len(allParmStart)
        tolIter = 1e-2
        from scipy import optimize
        x = optimize.minimize(specErrF, allParmStart, args=(self,parmActive,plot), tol=tolIter,
                              method="Nelder-Mead",options={'adaptive':True, 'disp':True, 'maxiter':maxIter})
        #print("complete:",x)
        
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
    
    
    def regrid(self,f_out,th_out):
            import numpy as np
            from scipy.interpolate import interp2d
            f_in = self.f.flatten()
            th_in = self.th.flatten()
            S_in = self.S
            
            th_in=np.hstack((th_in-360,th_in,th_in+360)).flatten()
            S_in=np.transpose(np.hstack((S_in,S_in,S_in)))

            interpolator = interp2d(f_in,th_in,S_in)
            S_out = np.transpose(interpolator(f_out,th_out))
            self.f = f_out
            self.th = th_out
            self.S = S_out
            self.autoCorrect()
            
            
    def findPeaks(self, sigmaFreqHz = 0.01, sigmaDirDeg = 5 ):
        # sigmaFreqHz - std dev of the Gaussian smoother in frequency dimension
        # sigmaDirDeg - std dev of the Gaussian smoother in direction dimension
        # 
        # df_smoothing: frequency [Hz] step used to create interpolated spectrum for smoothing
        # dth_smoothing:direction step [deg]used to create interpolated spectrum for smoothing
        #
        # Jason McConochie, 4 Nov 2022, Rev 1
        # 
        
        smFloorPercentMaxS=0.05 # the floor of the peak finding as a proportion of max Ssm(f,th)

        df_smoothing = 0.001  
        dth_smoothing = 10
        
        def interp_spec(S_in,f_in,th_in,f_out,th_out):
            import numpy as np
            from scipy.interpolate import interp2d
            f_in = f_in.flatten()
            th_in = th_in.flatten()

            th_in=np.hstack((th_in-360,th_in,th_in+360)).flatten()
            S_in=np.transpose(np.hstack((S_in,S_in,S_in)))

            interpolator = interp2d(f_in,th_in,S_in)
            S_out = np.transpose(interpolator(f_out,th_out))
            return S_out
        
        sigmaFreq_imUnits = sigmaFreqHz / df_smoothing
        sigmaDir_imUnits = sigmaDirDeg /dth_smoothing
        
        # Regrid the spectrum and extend the edges by wrapping the directions at each end
        import numpy as np
        #from wavespectra.core.utils import interp_spec
        imSpec = waveSpec()
        imSpec.f = np.arange(self.f[0],self.f[-1],df_smoothing)
        imSpec.th = np.arange(self.th[0],self.th[-1],dth_smoothing)
        imSpec.S = interp_spec(self.S, self.f, self.th, imSpec.f, imSpec.th)
        imSpec.autoCorrect()
        #import matplotlib.pyplot as plt
        #plt.pcolormesh(imSpec.f,imSpec.th,np.transpose(imSpec.S))
        
        from skimage import img_as_float
        im = img_as_float(imSpec.S)
        from skimage.filters import gaussian
        im = gaussian(im, sigma=(sigmaFreq_imUnits, sigmaDir_imUnits), truncate=3.5, channel_axis=2)
        from skimage.feature import peak_local_max
        #coordinates = peak_local_max(im, min_distance=1, threshold_abs=1e-3)
        coordinates = peak_local_max(im, min_distance=1, threshold_abs=smFloorPercentMaxS*np.amax(im),exclude_border=False)
        
        nCoord = len(coordinates)
        Tp = np.zeros(nCoord)
        ThetaP = np.zeros(nCoord)
        iTp = np.zeros(nCoord,int)
        iThetaP = np.zeros(nCoord,int)
        S = np.zeros(nCoord)
        Ssmooth = np.zeros(nCoord)
        for i,iPeak in enumerate(coordinates):
             Tp[i] = 1/imSpec.f[coordinates[i][0]]
             ThetaP[i] = imSpec.th[coordinates[i][1]]
             
             iTp[i] = np.argmin(np.abs(self.f - (1/Tp[i])))
             iThetaP[i] = np.argmin(np.abs(self.angDiff(self.th, ThetaP[i])))
             
             S[i] = imSpec.S[coordinates[i][0],coordinates[i][1]]
             Ssmooth[i] = im[coordinates[i][0],coordinates[i][1]]
         
        # Return also the smoothed spectrum
        #smSpec = waveSpec()
        #smSpec.f = self.f
        #smSpec.th = self.th
        #smSpec.S = im
        imSpec.S = im
        imSpec.autoCorrect()
         
        # Returns the Tp, ThetaP of the peaks and the smoothed spectrum smSpec
        # as a wvSpec object and the index of the peaks of Tp and Theta P
        #return Tp, ThetaP, S, imSpec, iTp_pk, iThetaP_pk, Ssmooth_pk  
        pks = {'Tp':Tp, 'ThetaP':ThetaP, 'Sp':S, 'iTp':iTp, 'iThetaP':iThetaP, 'Ssm':Ssmooth}
        return pks, imSpec 
    
    
    
    
    
    
    
    
              
    def processWindSeaSpec(self, wspd = None, wdir = None, dpt = None, agefac = 1.7, dirSprd = 1 ):
        '''
        Takes the spectrum returns a wind sea spectrum only containing wave frequencies and directions
        contained within the region defined as that with wave components with a wave age less then
        agefac in the direction of the wind direction (wdir) and for wind speed (wspd)
        
        wdir must be relative to N, coming from (or same as the wave direction convention used)
        
        The wind speed compoonent is made broader in direction space by using dirSprd expoenent
        on the cos function of the wind speed component (found to be more used for observations)
        
        If wdir is not give then the wind direction is estimated from the spectrum as the 
        peak direction of all spectral components with a frequency higher than 0.25 Hz
        
        Jason McConochie, 4 Nov 2022, Rev 1
        '''
        import numpy as np   
        import numpy.matlib as ml
        import copy
        wsSpec = copy.deepcopy(self)
        for i,v in enumerate(wsSpec.f):
            if wsSpec.f[i] < 0.25:
                wsSpec.S[i,:] = np.zeros(len(wsSpec.th))
        wsSpec.autoCorrect()
        wsIntPar = wsSpec.specIntParm()
        # Get wind sea direction
        if wdir == None:
            wdir = wsIntPar[6] # use ThetaP as wind direction
        
     
        # Make wind sea mask
        from wavespectra.core.utils import celerity
        D2R = np.pi/180
        dth = self.angDiff(self.th,wdir)
        wind_speed_component = agefac * wspd * np.cos(D2R*dth)**(dirSprd)
        wave_celerity = celerity(self.f, dpt)
        nth = len(self.th)
        nf = len(self.f)
        windseamask = np.transpose(ml.repmat(wave_celerity,nth,1)) <= ml.repmat(wind_speed_component,nf,1)
        idxThetaPSea = -1
        for iFreq,vFreq in enumerate(wsSpec.f):
            tMask = windseamask[iFreq,:]
            idx = np.where(tMask)
            if len(idx[0]) > 1:
                idxThetaPSea = int(np.mean(idx))  # need to vector mean
                break
        if idxThetaPSea == -1:
            # There is no wind sea
            return None, None, None
        else:
            ThetaPSea = wsSpec.th[idxThetaPSea]
        
        # Make wind sea spectrum alone masked by the wind sea mask
        wsSpec = copy.deepcopy(self)
        wsSpec.S = self.S * windseamask*1 
        wsSpec.autoCorrect()
        wsIntPar = wsSpec.specIntParm()
        TpSea = wsIntPar[1]
        ThetaPSea = wsIntPar[6]
        # find nearest index of wind sea
        iTp = np.argmin(np.abs(wsSpec.f - (1/TpSea)))
        iThetaP = np.argmin(np.abs(wsSpec.angDiff(wsSpec.th, ThetaPSea)))
        Sp = wsSpec.S[iTp,iThetaP]
 
        pks = {'Tp':TpSea, 'ThetaP':ThetaPSea, 'Sp':Sp, 'iTp':iTp, 'iThetaP':iThetaP}
        return pks, wsSpec, windseamask
    
    
    
    
    def reducePeaksClustering(self, pks, maxPeaks, plotClusterSpace = False, tag = "waveSpec"):
        '''
        Takes a set of 2D spectrum peak locations and clusters them together to reduce 
          the number of peak locations to the number requested.  
        
        Provde a dictionary pks, containing vectors length nPeaks of Tp[nPeaks] (seconds) and 
          ThetaP[nPeaks] (degrees) and smmothed spectrum densitory value at Tp,ThetaP abd the
          algorithm will select from that set of peaks no more than maxPeaks.
    
          pks['Tp'] = Tp_pk[nPeaks] (seconds)
          pks['ThetaP'] = ThetaP_pk[nPeaks] (degrees)
          pks['Ssm'] = Ssmooth_pk[nPeaks] (m^2/(Hz.deg)
          
        Algorithm will cluster the Tp, ThetaP space into clusters and select, within each
          cluster, one peak.  It takes the peak in each cluster with the largest S(f,th) spectral
          density.  Smooth_pk[nPeaks] should be a vector of the smoothed version of the 2D 
          spectral density at each of the Tp, ThetaP pairs.
        
        plotClusterSpace - if True will make a plot of the Tp-v-ThetaP space peaks as well
          as a plot of the normalised clustering space.
        tag - text string identifier pre-predended to filename of plot image saved to png.
          May include the path (e.g.  c:\myPathtoDir\tagName)
        
          
        The normalised clustering space is defined as Tp * cos(ThetaP), x1scale * Tp * sin(ThetaP)
           Default x1scale = 2.5. This helps bring more weight to Tp in the normalised space.
        
        Jason McConochie, 4 Nov 2022, Rev 1
        '''
        import numpy as np         
        
        # 0. Default for the scaling of the y-norm space
        x1Scale = 2.5
            
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
        if len(pks['Tp']) > maxPeaks:
            print(f"running reducePeaksClustering {len(pks['Tp'])},{maxPeaks}")
            
            # C1. Run kmeans clustering on all peaks 
            import os
            os.environ['OMP_NUM_THREADS'] = str(1)
            from sklearn.cluster import KMeans
            features = real2norm(pks['Tp'], pks['ThetaP'])
            nCl = np.min([len(pks['Tp']),maxPeaks])
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
            pks_sel = {}
            for key in pks.keys():
                pks_sel[key] = np.zeros(nClusters)
            pks_sel['idx'] = np.zeros(nClusters,int)
            for tClus in range(0,nClusters):
                idx_pks = np.ndarray.flatten(np.argwhere(tClus == whichClus) )
                # We use the smooth spectrum density as it give more information about the surrounds of the peak (ie.Hs)
                tClus_sortedS = np.argsort(pks['Ssm'][idx_pks])
                idx_pk = idx_pks[tClus_sortedS[-1]]
                pks_sel['Tp'][tClus] = pks['Tp'][idx_pk]
                pks_sel['ThetaP'][tClus] = pks['ThetaP'][idx_pk]
                pks_sel['idx'][tClus] = idx_pk
            useClustering = True
            
            # C3. Plot real and normalised cluster space - used for diagnosis only
            if plotClusterSpace:
                import matplotlib.pyplot as plt
                f,a = plt.subplots(1,2,figsize=(10,5))
                ta = a[0]
                ta.plot(pks['Tp'],pks['ThetaP'],'k.',ms=16)
                ta.plot(pks_sel['Tp'],pks_sel['ThetaP'],'m.',ms=8)
                for i in range(0,len(features[:,0])):
                    ta.text(pks['Tp'][i],pks['ThetaP'][i],str(whichClus[i])+f" ({pks['Tp'][i]:.1f},{pks['ThetaP'][i]:.0f},{pks['Ssm'][i]:.4f})",fontsize=10)
                ta.set_xlim([0,20])
                ta.set_ylim([0,360])
                ta.set_title('Real space Tp(x) ThetaP(y)')
                ta = a[1]
                ta.plot(features[:,0],features[:,1],'k.',ms=16)
                ta.plot(features[pks_sel['idx'],0],features[pks_sel['idx'],1],'m.',ms=8)
                ta.axis('square')
                for i in range(0,len(features[:,0])):
                    ta.text(features[i,0],features[i,1],str(whichClus[i])+f" ({pks['Tp'][i]:.1f},{pks['ThetaP'][i]:0.0f},{pks['Ssm'][i]:2.4f})",fontsize=10)
                ta.set_title('Normal/clustering space')
                f.savefig(f"{tag}_cs.png")
                plt.close(f)
             
         
        else:
            # D. Cluserting not required as supplied number of peaks is less than requested
            useClustering = False
            pks_sel = pks
            pks_sel['idx'] = list(range(0,len(pks_sel['Tp'])))
            whichClus = []
            
        
        return pks_sel, useClustering, whichClus
    
    
    
    
    
    
    def fit2DSpectrum(self,inConfig):  
        '''
        Parameters
        ----------
        inConfig : dictionary
            DESCRIPTION. 
            maxPartitions: integer number for the maximum number of spectral partitions to return
            useClustering: True or False
                # True: will take all peaks and select maxPartitions from the peaks using clustering 
                # False:  Takes largest peaks from smoothed spectrum  
            useWind: True or False
            useFittedWindSea: True or False
                 # True: TpSea, ThetaPSea calculated by fitting a spectrum to the spectra in the wind sea masked area
                 # False: TpSea, ThetaPSea from maximum S of the smoothed spectrum of all peaks in the wind sea mask area
            useWindSeaInClustering: True or False 
                 # True: Puts TpSea, ThetaPSea into the clustering and lets it take care of it.
                 # False: Do not use TpSea, ThetaPSea in clustering but instead ensures a wind sea spectrum
                         # is fitted in the final fitting as the first partition fixing TpSea, ThetaPSea    
            if useWind == True:
                REQUIRED:
                    wspd: wind speed in m/s    
                    dpt: water depth in m
                OPTIONAL:
                    wdir: wind direction [degN, from] (or same direction datum as spectrum), if
                        not provided wdir is taken as mean direction of the spectrum for S(f>0.25Hz)
            spreadType: 'parametric' (no other option implemented yet)
                This will use Ewans swell for Tp>9s and Ewans (1998) cos2s frequency dependent spreading for Tp<9s
            doPlot: True or False
                If True will make a plot of input spectrum, smoothed spectrum, peaks found and selected and clusters
            saveFigFilename: String filename to save spectrum plot
            plotClusterSpace: True or False
                If True will plot the cluster real and normalized space - used for diagnostics on clustering

        Returns
        -------
        None.
  
        
        '''
        DEFAULT_CONFIG = {
                'maxPartitions': 2,
                'useClustering': True,
                'useWind': False,
                'useFittedWindSea': False,
                'useWindSeaInClustering': True,
                'spreadType': 'parametric',  #only option so far
                'wspd': None,
                'wdir': None,
                'dpt': None,
                'plotClusterSpace': False,
                'doPlot': False,
                'saveFigFilename': "",
                'iTime': "",
                'fTime': ""
        }
        fConfig = {}
        for key in DEFAULT_CONFIG.keys():
            if key in inConfig.keys():   
                fConfig[key] = inConfig[key]
            else:
                fConfig[key] = DEFAULT_CONFIG[key]
        
    
        agefac = 1.7
        seaExp = 1.0 # use 0.2 to make the wind sea mask area directionally wider
      
        import numpy as np   
      
        # A. Find spectral peaks
        pks, smSpec = self.findPeaks()
        if len(pks) == 0:
            vPart = [None]*8
            fitStatus =[False,None,None]
            print('fit2DSpectrum: No peaks found - check input spectrum')
            return vPart, fitStatus
        
        # B. Take advantage of the wind speed, direction and water depth if requested
        if fConfig['useWind']:
        
            # B1. Process the wind sea components
            pkWS, wsSpec, wsMask = self.processWindSeaSpec(fConfig['wspd'], fConfig['wdir'], fConfig['dpt'], agefac, seaExp)
            if pkWS == None:
                # No wind sea found
                fConfig['useWind'] = False 
                print('fit2DSpectrum: No wind sea found')
            else:    
                # B2. Get the wind sea masked area Tp/ThetaP
                if fConfig['useFittedWindSea']:
                    # Get Wind Sea Tp/ThetaP from nonlinear fitting to wind sea masked area
                    parmActive=[[True,True,True,0.07,0.09,-5,True,True]]
                    parmStart=[[2,6,3.3,0.07,0.09,-5,180,6]]
                    vWindSea, fitStatusWindSea = wsSpec.fitMulitDirJONSWAPCos2s(parmActive,parmStart,True,fConfig['spreadType'])
                    TpWindSea = vWindSea[0][1]
                    ThetaPWindSea = vWindSea[0][6]
                else:    
                    # Get Wind Sea Tp/ThetaP from maximum Ssm of all peaks in the wind sea mask area
                    TpWindSea = None
                    ThetaPWindSea = None
                    SsmMaxWindSea = 0
                    for iPeak,tPeak in enumerate(pks['iTp']):
                        if wsMask[pks['iTp'][iPeak],pks['iThetaP'][iPeak]]:
                            tSsmMaxWindSea = smSpec.S[pks['iTp'][iPeak],pks['iThetaP'][iPeak]] 
                            if tSsmMaxWindSea > SsmMaxWindSea:
                                TpWindSea = pks['Tp'][iPeak]
                                ThetaPWindSea = pks['ThetaP'][iPeak]   
                                SsmMaxWindSea = tSsmMaxWindSea
                 
                
                if TpWindSea == None:
                   # No wind sea found
                   fConfig['useWind'] = False 
                   print('fit2DSpectrum: No wind sea found')
                else:   
                    
                   # B3. Remove all peaks within windsea mask array 
                   iPeak = 0
                   while 1:
                       if wsMask[pks['iTp'][iPeak],pks['iThetaP'][iPeak]]:
                           # Remove peak
                           for key in pks.keys():
                               pks[key] = np.delete(pks[key], iPeak)
                       else:
                           iPeak += 1
                       if iPeak == len(pks['iTp']): break
                   
                   # B4. If using wind sea in cluster add to pks array for consideration
                   if fConfig['useWindSeaInClustering']:
                           pks['Tp'] = np.append(pks['Tp'],TpWindSea)
                           pks['ThetaP'] = np.append(pks['ThetaP'],ThetaPWindSea)
                           pks['iTp'] = np.append(pks['iTp'],np.argmin(np.abs(self.f - (1/TpWindSea))))
                           pks['iThetaP'] = np.append(pks['iThetaP'],np.argmin(np.abs(self.angDiff(self.th, ThetaPWindSea))))
                           pks['Sp'] = np.append(pks['Sp'],self.S[pks['iTp'][-1],pks['iThetaP'][-1]] )
                           pks['Ssm'] = np.append(pks['Ssm'],smSpec.S[pks['iTp'][-1],pks['iThetaP'][-1]])
                   else:
                       None
                       # Dont touch the peaks array, but later add in the wind sea as the first partition
                    
        
        def selectTopPeaks(nPeaksToSelect, pks):
            # Select peaks to keep 
            iS_pk = np.argsort(pks['Sp'])
            nPeaks = np.min([len(iS_pk),nPeaksToSelect])
            pks_sel = {}
            for key in pks.keys():
                pks_sel[key] = np.zeros(nPeaks)
            for i in range(1,nPeaks+1):
                itPk = iS_pk[-i]
                for key in pks.keys():
                    pks_sel[key][i-1] = pks[key][itPk]
            return pks_sel
        
        # C. Run the clustering or use highest peaks 
        nPeaksToSelect = fConfig['maxPartitions']
        if fConfig['useWind']:
            if fConfig['useClustering']:
                if not fConfig['useWindSeaInClustering']: 
                    nPeaksToSelect = nPeaksToSelect - 1
            #if fConfig['useFittedWindSea']:
            #    if not fConfig['useWindSeaInClustering']: 
            #        nPeaksToSelect = nPeaksToSelect - 1
                #else:
                #     vPart = [None] *8
                #     fitStatus =[False,None,None]
                #     print('fit2DSpectrum: Must have clustering selected to useWindSeaInClustering')
                #     return vPart, fitStatus
                
                
        if fConfig['useClustering']:
            pks_sel, fConfig['useClustering'], whichClus = \
                smSpec.reducePeaksClustering(pks, nPeaksToSelect, plotClusterSpace = fConfig['plotClusterSpace'], tag = fConfig['saveFigFilename'])
            if len(pks_sel['Tp']) == 0:
                pks_sel = selectTopPeaks(nPeaksToSelect,pks)
        else:
            pks_sel = selectTopPeaks(nPeaksToSelect,pks)    
        
        # D. Run the main fitting of the spectrum
        if len(pks_sel['Tp']) > 0:
            #[Hs,Tp,Gamma,sigmaa,sigmab,Exponent,WaveDir,sSpread] 
            parmActive = []
            parmStart = []
            if fConfig['useWind']:
                if not fConfig['useWindSeaInClustering']:  
                    parmActive.append([True,TpWindSea,True,0.07,0.09,True,ThetaPWindSea,TpWindSea])
                    parmStart.append([2,TpWindSea,3.3,0.07,0.09,-5,ThetaPWindSea,TpWindSea])
                    #pks_sel = selectTopPeaks(fConfig['maxPartitions']-1,pks) 
            for i in range(0,len(pks_sel['Tp'])):
                parmActive.append([True,pks_sel['Tp'][i],True,0.07,0.09,True,pks_sel['ThetaP'][i],pks_sel['Tp'][i]])
                parmStart.append([2,pks_sel['Tp'][i],3.3,0.07,0.09,-5,pks_sel['ThetaP'][i],pks_sel['Tp'][i]])
            
            vPart,fitStatus = self.fitMulitDirJONSWAPCos2s(parmActive,parmStart,True,fConfig['spreadType'])
            # Sort the partitions in order of increasing Tp
            partTps = [x[1] for x in vPart]
            sPartTps = np.argsort(partTps)
            nPart = [[]]*len(vPart)
            for iPart,tPart in enumerate(vPart):
                nPart[iPart] = vPart[sPartTps[iPart]] 
            vPart = nPart
        else:
            print("*** Problem should not get here")
        
        # E. Reconstruct the fitted spectra
        sTot = waveSpec()
        sTot.f = np.array(self.f)
        sTot.th = np.array(self.th)
        sTot.S = self.S * 0
        sTot.autoCorrect()
        for iPart,tPart in enumerate(vPart):
            s = waveSpec()
            s.f = np.array(self.f)
            s.th = np.array(self.th)
            s.makeJONSWAP2D(tPart[0:6],tPart[6:9],fConfig['spreadType'])  
            s.autoCorrect()
            sTot.S = sTot.S + s.S
        ft = sTot
        ft.autoCorrect()
        


        if fConfig['doPlot']:
            import matplotlib.pyplot as plt

            #. F Make plots of the spectra original and fitted
            def plotPeaks(ax):
                for i in range(0,len(pks['Tp'])):
                    ax.plot(1/pks['Tp'][i],pks['ThetaP'][i],'w.',ms=16)
                    ax.plot(1/pks['Tp'][i],pks['ThetaP'][i],'m.',ms=10)
                    if fConfig['useClustering']:
                        ax.text(1/pks['Tp'][i],pks['ThetaP'][i],str(whichClus[i]),color='white',fontsize=22,ha="left")
                o = ""
                for i in range(0,len(pks_sel['Tp'])):
                    ax.plot(1/pks_sel['Tp'][i],pks_sel['ThetaP'][i],'w.',ms=16)
                    ax.plot(1/pks_sel['Tp'][i],pks_sel['ThetaP'][i],'k.',ms=10)
                    o += f"{pks_sel['Tp'][i]:0.1f}-{pks_sel['ThetaP'][i]:0.0f}    "
                ax.text(0,0,"       "+o,color='w',size=12)
                    
            S = self.S; S[S<1e-9]=1e-9;
            Sm = smSpec.S; Sm[Sm<1e-9]=1e-9;
            f,b = plt.subplots(3,3,figsize=(15,15))
            ta = b[0][0]
            ta.pcolormesh(self.f,self.th,np.log(np.transpose(S+1e-9)),clim=[-15,0]) 
            if fConfig['useWind']: ta.contour(wsSpec.f,wsSpec.th,np.transpose(wsMask),levels=[0.5],colors='white')
            plotPeaks(ta)
            ta.set_title(f"Input@{fConfig['fTime']},{fConfig['iTime']}")
            ta = b[0][1]
            ta.pcolormesh(smSpec.f,smSpec.th,np.log(np.transpose(Sm+1e-9)),clim=[-15,0]) 
            ta.set_title('Smoothed spectrum')
            if fConfig['useWind']: ta.contour(wsSpec.f,wsSpec.th,np.transpose(wsMask),levels=[0.5],colors='white')
            plotPeaks(ta)
            ta = b[0][2]
            ta.pcolormesh(ft.f,ft.th,np.log(np.transpose(ft.S+1e-9)),clim=[-15,0]) 
            ta.set_title('Reconstructed spectrum')
            if fConfig['useWind']: ta.contour(wsSpec.f,wsSpec.th,np.transpose(wsMask),levels=[0.5],colors='white')
            plotPeaks(ta)
            
            ta = b[1][0]
            ta.pcolormesh(self.f,self.th,(np.transpose(self.S))) 
            if fConfig['useWind']: ta.contour(wsSpec.f,wsSpec.th,np.transpose(wsMask),levels=[0.5],colors='white')
            plotPeaks(ta)
            ta = b[1][1]
            ta.pcolormesh(smSpec.f,smSpec.th,(np.transpose(smSpec.S))) 
            ta.set_title('Smoothed spectrum')
            if fConfig['useWind']: ta.contour(wsSpec.f,wsSpec.th,np.transpose(wsMask),levels=[0.5],colors='white')
            plotPeaks(ta)
            ta = b[1][2]
            ta.pcolormesh(ft.f,ft.th,(np.transpose(ft.S))) 
            ta.set_title('Reconstructed spectrum')
            if fConfig['useWind']: ta.contour(wsSpec.f,wsSpec.th,np.transpose(wsMask),levels=[0.5],colors='white')
            plotPeaks(ta)
            
            ta = b[2][0]
            ta.plot(self.f,self.Sf,'k-',label="Input")
            ta.plot(ft.f,ft.Sf,'b-',label="Reconstructed")
            ta.legend()
            ta.set_title('Frequency spectrum')
            ta = b[2][1]
            ta.plot(self.th,self.Sth,'k-',label="Input")
            ta.plot(ft.th,ft.Sth,'b-',label="Reconstructed")
            ta.set_title('Direction spectrum')
            ta = b[2][2]
            ta.plot(self.f,self.Sf,'k-',label="Input")
            ta.plot(ft.f,ft.Sf,'b-',label="Reconstructed")
            ta.set_xscale('log')
            ta.set_yscale('log')
            ta.set_ylim([1e-3,1e3])
            ta.set_title('Frequency spectrum')
            plt.grid()
               
            f.savefig(fConfig['saveFigFilename']+"_pk.png")
            plt.close(f)
            
        
        return vPart, fitStatus