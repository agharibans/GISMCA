import matplotlib.pyplot as plt
import numpy as np
import os
from GISMCA import __version__

def plotOverall(time,data,start,end,pks,timeTTX,filename):
    '''
    Saves a figure that shows the data with identified peaks.

    Parameters
    ----------
    time : array_like
        Time vector in seconds.
    data : array_like
        Filtered tissue contraction data in milliNewtons.
    start : float
        Start time for analysis in seconds.
    end : float
        End time for analysis in seconds.
    pks : array_like
        Identified location of peaks for each contraction.
    timeTTX : float
        Time of stimulus in seconds.
    filename : string
        Name of the file for saving the figure.

    Returns
    -------
    None
    '''

    #create directory for saving figure
    if os.path.exists('./'+filename)==False:
        os.mkdir(filename)

    fig,ax = plt.subplots(figsize=(time[-1]/60/5,np.nanmax(data)/5))
    ax.plot(time/60,data,zorder=1,linewidth=1)
    ax.scatter(time[pks[np.isfinite(pks)].astype(int)]/60,data[pks[np.isfinite(pks)].astype(int)],
               color='#d62728',s=10,zorder=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Force (mN)')
    ax.set_xticks(np.arange(0,time[-1]/60,10))
    ax.set_xlim([start/60,end/60])
    ax.set_yticks(np.arange(0,np.nanmax(data),10))
    ax.set_ylim([0,np.nanmax(data)])
    ax.set_title(filename)
    ax.grid(True)
    if timeTTX!=None: ax.axvline(timeTTX/60,color='black')
    ax.set_axisbelow(True)
    fig.tight_layout()
    
    plt.savefig('./'+filename+'/'+filename+' - overall - v'+__version__+'.jpg',
                dpi=200,quality=80,optimize=True,bbox_inches='tight')
    
    plt.close()


def plotAll(time,data,fs,pks,left,right,featuresDF,filename):
    '''
    Save figures for each contraction with features.

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
    featuresDF : DataFrame
		Pandas dataframe with all contraction features.
    filename : string
        Name of the file for saving the figure.

    Returns
    -------
    None
    '''

    #create directory for saving figure
    if os.path.exists('./'+filename)==False:
        os.mkdir(filename)

    maxHeight = 10# np.max(featuresDF['Height (mN)'])
    maxDuration = np.max(featuresDF['Duration - Total (s)'])
    plot = 1
    for ind in range(0,len(pks)):
        ii = np.mod(int(ind/10),10)
        jj = np.mod(ind,10)
        if (ii==0)&(jj==0): fig,ax = plt.subplots(nrows=10,ncols=10,figsize=(50,50))

        if np.isnan(pks[ind])==False:
            start = int(left[ind])
            end = int(right[ind])
            peak = int(pks[ind])
            height = np.round(data[peak]-np.min([data[start],data[end]]),2)

            dataWin = data[start-(1*fs):end+1+(1*fs)]
            timeWin = np.arange(0,len(dataWin)/fs,1/fs)-1
            peak -= (start-(1*fs))
            end -= (start-(1*fs))
            start = 1*fs

            #save plot
            ax[ii,jj].plot(timeWin,dataWin)
            ax[ii,jj].scatter(timeWin[peak],dataWin[peak])
            ax[ii,jj].scatter(timeWin[start],dataWin[start])
            ax[ii,jj].scatter(timeWin[end],dataWin[end])
            ax[ii,jj].vlines(x=timeWin[peak], ymin=dataWin[peak]-height, ymax=dataWin[peak])
            ax[ii,jj].hlines(y=dataWin[peak]-height, xmin=timeWin[start], xmax=timeWin[end])
            ax[ii,jj].set_xlabel('Time (s)')
            ax[ii,jj].set_ylabel('Force (mN)')
            ax[ii,jj].set_title('Peak #'+str(ind+1)+' - '+str(np.round(time[int(left[ind])]/60,1))+' min')
            if height<=maxHeight:
                ax[ii,jj].set_ylim(top=np.min([dataWin[start],dataWin[end]])+maxHeight+1)
            ax[ii,jj].set_xlim(right = timeWin[start]+maxDuration)

        else: #save black box for event marker
            ax[ii,jj].fill_between([0,1],0,1,color='grey')
            ax[ii,jj].set_xlim([0,1])
            ax[ii,jj].set_ylim([0,1])
            ax[ii,jj].set_axis_off()

        if (ii==9)&(jj==9) or (ind+1==len(pks)):
            fig.tight_layout()
            plt.savefig('./'+filename+'/'+filename+' - '+str(plot)+' - v'+__version__+'.jpg',
                        dpi=100,quality=80,optimize=True,bbox_inches='tight')
            plot+=1
            plt.close()


def plotTTX(data,fs,pks,timeTTX,filename,timePre=10,timePost=18):
    '''
    Save figure for the mean contraction before and after stimulus.

    Parameters
    ----------
    data : array_like
        Filtered tissue contraction data in milliNewtons.
    fs : int
        Sampling frequency.
    pks : array_like
        Identified location of peaks for each contraction.
    timeTTX : float
        Time of stimulus in seconds.
    filename : string
        Name of the file for saving the figure.
    timePre : float
        Time in seconds before the peak to plot (default 10s).
    timePost : float
        Time in seconds after the peak to plot (default 18s).

    Returns
    -------
    None
    '''

    pkPostTTX = np.where(pks>int(timeTTX*fs))[0][0]

    pre = []
    post = []
    for ind in range(0,len(pks)-1):
        if np.isnan(pks[ind])==False:
            samples = np.arange(int(pks[ind]-timePre*fs),int(pks[ind]+timePost*fs))
            if ind < pkPostTTX:
                pre.append(data[samples])
            if ind > pkPostTTX:
                post.append(data[samples])

    pre = np.vstack(pre)
    post = np.vstack(post)

    #mean centering
    pre -= np.mean(pre,axis=1, keepdims=True)
    post -= np.mean(post,axis=1, keepdims=True)

    #plot
    fig,ax = plt.subplots(ncols=2,figsize=(7,3.5),sharey=True)

    tPlot = np.arange(-timePre*fs,timePost*fs)/fs

    meanPre = np.nanmean(pre,axis=0) - np.nanmean(pre,axis=0)[0]
    meanPost = np.nanmean(post,axis=0) - np.nanmean(post,axis=0)[0]

    ax[0].plot(tPlot,meanPre)
    ax[0].fill_between(tPlot,meanPre-np.nanstd(pre,axis=0),meanPre+np.nanstd(pre,axis=0),alpha=0.15)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Force (mN)')
    ax[0].set_title('Pre')
    ax[0].set_xlim([tPlot[0],tPlot[-1]])

    ax[1].plot(tPlot,meanPost)
    ax[1].fill_between(tPlot,meanPost-np.nanstd(post,axis=0),meanPost+np.nanstd(post,axis=0),alpha=0.15)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title('Post')
    ax[1].set_xlim([tPlot[0],tPlot[-1]])

    fig.tight_layout()
    plt.savefig('./'+filename+'/'+filename+' - pre-post - v'+__version__+'.jpg',
                dpi=300,quality=80,optimize=True,bbox_inches='tight')
    
    plt.close()