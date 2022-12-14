% Example mat file shows how the matlab file should be created
% for use with fitFromMatlabFile.py
%
% Variables required in mat file:
%   td[nTimes,6] - matrix of date vector each row contains year, month, day, hour, minute, second 
%   fd[nFre] - vector of spectral frequencies in Hz
%   thetad[nDir] - vector of spectral directions in deg
%   spec2d[nTimes, nFre, nDir] - matrix of spectral densities S(f,th) in
%   m^2/(Hz.deg)
% 
% 

load('data/ExampleWaveSpectraObservations.mat')
whos
unix('pip3 install wavespectra2dsplitfit')
unix('python3 fitFromMatlabFile.py data/ExampleWaveSpectraObservations.mat')
res = readtable('fittedParms.csv')
