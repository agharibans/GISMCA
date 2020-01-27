import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.optimize import curve_fit
import pandas as pd
import os
from GISMCA import plot

def loadData(filename):
    '''
    Function to import data in .mat format exported by Biopac's Acknowledge software.
    
    Parameters
    ----------
    filename : str
        Path to .mat file.
    
    Returns
    -------
    data : array_like
        Tissue contraction data in milliNewtons.
    time : array_like
        Time vector in seconds.
    fs : int
        Sampling frequency.

    '''
    
    data = loadmat(filename)
    fs = int(1/(data['isi'][0][0]/1000)) #sampline frequency
    data = data['data'][:,0]*9.8 #convert from grams to milliNewtons

    #decimate
    if fs==100:
        data = signal.decimate(data,4,n=499,ftype='fir')
        data = signal.decimate(data,5,n=499,ftype='fir')
        fs = 5

    time = np.arange(0,data.shape[0]/fs,1/fs)

    return data, time, fs


def filterData(data,fs):
    '''
    Apply a low-pass filter (fc = 0.5 Hz) to the data to remove high-frequency noise.

    Paramters
    ---------
    data : array_like
        Tissue contraction data in milliNewtons.
    fs : int
        Sampling frequency.

    Returns
    -------
    dataFilt : array_like
        Filtered tissue contraction data in milliNewtons.

    '''

    #low pass filter
    b = signal.firwin(399,.5,fs=fs,window=('kaiser',6.0),pass_zero=True)
    w, h = signal.freqz(b)
    dataFilt = signal.filtfilt(b,[1],x=data,axis=0)

    return dataFilt


def findPeaks(time,data,fs,start,end,minProminence=0.05*9.8):
    '''
    Find all the peaks using the scipy function find_peaks.  The peaks have to be
    at least 10 seconds appart, the width of the peak has to be at least 5 seconds,
    and with a minimum specified amplitude (default 0.05).  The peak locations 
    along with the left and right points at half of the amplitude are saved.

    Parameters
    ----------
    time : array_like
        Time vector in seconds.
    data : array_like
        Filtered tissue contraction data in milliNewtons.
    fs : int
        Sampling frequency.
    start : float
        Start time for analysis in seconds.
    end : float
        End time for analysis in seconds.
    minProminence : float
        Minimum amplitude to be considered a contraction (default 0.05 g).

    Returns
    -------
    pks : array_like
        Identified location of peaks that meet the criteria.
    left : array_like
        Left location corresponding to each peak at half of the amplitude.
    right : array_like
        Right location corresponding to each peak at half of the amplitude.
    '''
    
    data[(time<start)|(time>end)] = np.nan
    
    pks = signal.find_peaks(data,distance=10*fs,prominence=minProminence,
                            wlen=30*fs,width=5*fs,rel_height=0.98)
    left = (pks[1]['left_ips']).astype('int')
    right = (pks[1]['right_ips']).astype('int')
    pks = pks[0]

    #calculate area of each contraction
    area = np.zeros(len(pks))
    for ii in range(0,len(pks)):
        area[ii] = np.trapz(data[left[ii]:right[ii]]-data[left[ii]],dx=1/fs)

    #remove contractions that are too small
    left = left[area>=3]
    right = right[area>=3]
    pks = pks[area>=3]

    return pks, left, right


def calculateFeatures(time,data,fs,pks,left,right,timeTTX,filename):
    '''
    Calculate features for each detected contraction and save as csv file.

    Parameters
    ----------
    time : array_like
        Time vector in seconds.
    data : array_like
        Filtered tissue contraction data in milliNewtons.
    fs : int
        Sampling frequency.
    pks : array_like
        Identified location of peaks for each contraction.
    left : array_like
        Left location corresponding to each peak at half of the amplitude.
    right : array_like
        Right location corresponding to each peak at half of the amplitude.
    timeTTX : float
        Time of stimulus in seconds.
    filename : string
        Name of the file for saving csv.

    Returns
    -------
    featuresDF : DataFrame
        Pandas DataFrame with features for each peak.
    '''

    #find the peak after vasopressin and insert nan before
    if np.isnan(timeTTX)==False:
        pkPostTTX = np.where(pks>int(timeTTX*fs))[0][0]
        left = np.hstack([left[:pkPostTTX],np.nan,left[pkPostTTX:]])
        right = np.hstack([right[:pkPostTTX],np.nan,right[pkPostTTX:]])
        pks = np.hstack([pks[:pkPostTTX],np.nan,pks[pkPostTTX:]])

    #define function for exponential fit
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    featuresDF = pd.DataFrame()

    for ind in range(0,len(pks)):
        features = {}
        features['Peak #'] = ind+1
        
        if np.isnan(pks[ind])==False:
            start = int(left[ind])
            end = int(right[ind])
            peak = int(pks[ind])
            height = np.round(data[peak]-np.min([data[start],data[end]]),2)

            features['Height (mN)'] = height
            features['Amplitude - Start (mN)'] = np.round(data[start],2)
            features['Amplitude - Peak (mN)'] = np.round(data[peak],2)
            features['Amplitude - End (mN)'] = np.round(data[end],2)
            features['Time - Start (s)'] = np.round(time[start],1)
            features['Time - Peak (s)'] = np.round(time[peak],1)
            features['Time - End (s)'] = np.round(time[end],1)
            features['Duration - Total (s)'] = np.round(features['Time - End (s)'] - features['Time - Start (s)'],1)
            features['Duration - Onset (s)'] = np.round(features['Time - Peak (s)']-features['Time - Start (s)'],1)
            features['Duration - Decay (s)'] = np.round(features['Time - End (s)']-features['Time - Peak (s)'],1)
            features['Rate - Mean Onset (mN/s)'] = np.round(np.mean(np.diff(data[start:peak+1]-data[start])/(1/fs)),3)
            features['Rate - Mean Decay (mN/s)'] = np.round(np.mean(np.diff(data[peak:end]-data[start])/(1/fs)),3)
            if end-peak>2:
                popt,_ = curve_fit(func, np.arange(0,(end-peak)/fs,1/fs), data[peak:end],p0=[1,0,1],maxfev=5000)
                features['Rate - Exp Decay Time Constant'] = -np.round(popt[1],3)
            features['Area - Onset (mN-s)'] = np.round(np.trapz(data[start:peak+1]-data[start],dx=1/fs),2)
            features['Area - Decay (mN-s)'] = np.round(np.trapz(data[peak:end]-data[start],dx=1/fs),2)
            features['Area - Total (mN-s)'] = np.round(np.trapz(data[start:end]-data[start],dx=1/fs),2)

        featuresDF = featuresDF.append(features,ignore_index=True)

    featuresDF['Peak #'] = featuresDF['Peak #'].astype('int')
    featuresDF['Time To Next Peak (s)'] = np.append(np.diff(featuresDF['Time - Peak (s)']),np.nan)
    featuresDF = featuresDF.set_index('Peak #')
    featuresDF = featuresDF[['Time - Start (s)','Time - Peak (s)','Time - End (s)','Time To Next Peak (s)',
                            'Duration - Onset (s)','Duration - Decay (s)','Duration - Total (s)',
                            'Amplitude - Start (mN)','Amplitude - Peak (mN)','Amplitude - End (mN)',
                            'Height (mN)','Area - Onset (mN-s)','Area - Decay (mN-s)','Area - Total (mN-s)',
                            'Rate - Mean Onset (mN/s)','Rate - Mean Decay (mN/s)','Rate - Exp Decay Time Constant']]

    #create directory for saving figure
    if os.path.exists('./'+filename)==False:
        os.mkdir(filename)

    #save CSV
    featuresDF.to_csv('./'+filename+'/'+filename+' - features.csv')

    return featuresDF


def runAll(filename,start,end,timeTTX,minProminence=0.05*9.8):
    '''
    Run entire analysis and plotting pipeline for analyzing in vitro gastrointestinal
    smooth muscle contraction data.  This function saves figures and data in a directory
    with the same name as 'filename'.

    Parameters
    ----------
    filename : str
        Path to .mat file.
    start : float
        Start time for analysis in seconds.
    end : float
        End time for analysis in seconds.
    timeTTX : float
        Time of stimulus in seconds.
    minProminence : float
        Minimum amplitude to be considered a contraction (default 0.05 g).
    
    Returns
    -------
    None

    '''

    data, time, fs = loadData(filename+'.mat')
    data = filterData(data,fs)
    pks,left,right = findPeaks(time,data,fs,start,end,minProminence)
    featuresDF = calculateFeatures(time,data,fs,pks,left,right,timeTTX,filename)
    plot.plotOverall(time,data,start,end,pks,timeTTX,filename)
    plot.plotAll(time,data,fs,pks,left,right,featuresDF,filename)
    plot.plotTTX(data,fs,pks,timeTTX,filename,timePre=10,timePost=18)
