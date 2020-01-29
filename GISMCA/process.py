import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.optimize import curve_fit
import pandas as pd
import os
from GISMCA import plot
from GISMCA import __version__


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
    fs = int(1/(data['isi'][0][0]/1000)) #sampling frequency
    data = data['data']*9.8 #convert from grams to milliNewtons

    if data.shape[0]<data.shape[1]:
    	data = data.T

    #decimate
    if fs==100:
        data = signal.decimate(data,4,n=499,ftype='fir',axis=0)
        data = signal.decimate(data,5,n=499,ftype='fir',axis=0)
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


def findPeaks(time,data,fs,start,end,events,minHeight=0.05*9.8,minDistance=10):
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
    events : list
        Times of events in seconds.
    minHeight : float
        Minimum height to be considered a contraction (default 0.05 g).
    minDistance : float
    	Minimum time in seconds between peaks (default 10 seconds).

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
    
    pks = signal.find_peaks(data,distance=minDistance*fs,prominence=minHeight,
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

    #find the peak after events and insert nan before
    for event in events:
        if event!=None:
            pkPostEvent = np.where(pks>int(event*fs))[0][0]
            left = np.hstack([left[:pkPostEvent],np.nan,left[pkPostEvent:]])
            right = np.hstack([right[:pkPostEvent],np.nan,right[pkPostEvent:]])
            pks = np.hstack([pks[:pkPostEvent],np.nan,pks[pkPostEvent:]])

    return pks, left, right


def calculateFeatures(time,data,fs,pks,left,right,filename):
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
    filename : string
        Name of the file for saving csv.

    Returns
    -------
    featuresDF : DataFrame
        Pandas DataFrame with features for each peak.
    '''

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
    featuresDF['Time To Next Peak (s)'] = np.append(np.round(np.diff(featuresDF['Time - Peak (s)']),1),np.nan)
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
    featuresDF.to_csv('./'+filename+'/'+filename+' - features - v'+__version__+'.csv')

    return featuresDF


def runAll(filename,start,end,events,minHeight=0.05*9.8,minDistance=10):
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
    events : list
        Times of events in seconds.
    minHeight : float
        Minimum height to be considered a contraction (default 0.05 g).
    minDistance : float
    	Minimum time in seconds between peaks (default 10 seconds).
    
    Returns
    -------
    None

    Raises
    ------
    Exception
        If specified end time is after the end of the data.

    '''

    data, time, fs = loadData(filename+'.mat')
    data = filterData(data,fs)

    if end > data.shape[0]/fs:
        raise Exception('Specified end time greater than recording duration.')

    for ch in range(0,data.shape[1]):
        filenameCh = filename + ' - '+str(ch+1)
        pks,left,right = findPeaks(time,data[:,ch],fs,start,end,events,minHeight,minDistance)
        featuresDF = calculateFeatures(time,data[:,ch],fs,pks,left,right,filenameCh)
        plot.plotOverall(time,data[:,ch],start,end,pks,events,filenameCh)
        plot.plotAll(time,data[:,ch],fs,pks,left,right,featuresDF,filenameCh)
        plot.plotEvents(data[:,ch],fs,pks,events,filenameCh,featuresDF)