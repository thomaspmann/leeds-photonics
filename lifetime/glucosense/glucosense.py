# Lifetime functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    
def getFiles(folder='', ext='MAT'):
    """ 
    Function to get all matlab file names inside Data folder in pwd.
    
    The argument, folder, selects a folder inside the Data folder.
    
    """
    import glob as gb
    # Get list of files
    if folder == '':
        return gb.glob('./Data/*.%s' % ext)
    else:
        return gb.glob('./Data/%s/*.%s' % (folder, ext))


def getLifetime(time, intensity, sample, name, rejectTimeStart, rejectTimeEnd, 
                plotFit=True, saveFig=False, normalise=True):

    from scipy.optimize import curve_fit
    
    def model_func(t, a, b, c):
        return a*np.exp(-t/b)+c
    
    ## Signal preparation
    ind = np.where((time >= rejectTimeStart) & (time < max(time) - rejectTimeEnd))
    intensity = intensity[ind]
    time = time[ind]
    
    intensity = intensity - min(intensity)

    if normalise:
        intensity = intensity/max(intensity)

    try:
        guess = [max(intensity)-min(intensity), 10, min(intensity)]  # Guess for a, b, c coefficients
        popt, pcov = curve_fit(model_func, time, intensity, guess)   # Fit using Levenberg-Marquardt algorithm
        perr = np.sqrt(np.diag(pcov))                                # Error in coefficients to 1 std.
        
        tau = popt[1]         # Lifetime (ms)
        tauErr = perr[1]      # Error in lifetime
        
        # Chi Square Test
        from scipy.stats import chisquare
        chi2 = chisquare(intensity, f_exp=model_func(time, *popt))
        
        # Do plots
        f, (ax1, ax2, ax3) = plt.subplots(3,figsize=(15,15), sharex=False)
        ax1.set_title('Sample: %s. Lifetime = %.3f $\pm$ %.4f ms. $\chi^2$ = %.2f.' 
                      % (sample, tau, tauErr, chi2[0]), fontsize=16)

        ax1.set_ylabel('Intensity (A.U.)')
        ax1.plot(time, intensity, 'k.', label="Original Noised Data")
        ax1.plot(time, model_func(time, *popt), 'r-', label="Fitted Curve")
        ax1.axvline(tau,color='blue')
        ax1.set_xlim(0,max(time))
        ax1.legend()

        residuals = intensity - model_func(time, *popt)
        standd = np.std(residuals)
        ax2.set_ylabel('Residuals')
        ax2.plot(time, residuals)
        ax2.set_xlim(0,max(time))

        ax3.set_ylabel('Intensity (A.U.)')
        ax3.semilogy(time, intensity, 'k.', label="Original Noised Data")
        ax3.semilogy(time, model_func(time, *popt), 'r-', label="Fitted Curve")
        if normalise:
            ax3.set_ylim(10**-4,1)
        ax3.set_xlim(0,max(time))
        ax3.legend()
        
        if plotFit:
            #plt.show(block=True)
            plt.show()
            
        if saveFig:
            saveName = './Figs/' + name + '.png'
            print('... saving... ', end="")
            f.savefig(saveName, bbox_inches='tight', dpi=300)
            print('saved!')
        plt.close(f)
        
        return tau, tauErr, chi2
    
    except:
        print(f + ' did not fit a single exp :(')
        return np.NaN, np.NAN, np.NAN


def plotDecays(folder='', logarithm=True, saveFig=False):
    
    f = plt.figure()
    
    for file in getFiles(folder):
        spMatFile = loadmat(file)
        
        # Signal
        time = spMatFile['t']
        intensity = spMatFile['buffer_a_mv_mean']
        
        ## Signal preparation
        ind = np.where((time >= 1) & (time < max(time) - 60))
        intensity = intensity[ind]
        time = time[ind]
        
        intensity = intensity - min(intensity)
        #intensity = intensity / max(intensity)
        
        
        if not logarithm:
            plt.plot(time, intensity, label=spMatFile['info']['laserPulseWidth'])
        else:
            plt.semilogy(time, intensity, label=spMatFile['info']['laserPulseWidth'])
            plt.xlim(0,50)
            #plt.ylim(10**-4,10)
        plt.pause(0.1)

    plt.legend()
    plt.show()
    
    if saveFig:
        print('... saving... ', end="")
        savename = './Figs/' + spMatFile['info']['sample'] + '.png'
        f.savefig(savename, bbox_inches='tight', dpi=300)
        print('saved!')
        plt.close(f)

        
def plotConcentration(concentration, lifetime, standd, folder):
    fig = plt.figure()
    ax = fig.add_axes([1,1,1,1])
    ax.errorbar(concentration, lifetime, standd,  linestyle='',marker='^')

    ax.ticklabel_format(useOffset=False)
    plt.title(folder)
    plt.xlabel('Concentration (mg/dl)')
    plt.ylabel('Lifetime (ms)')
    saveName = './Figs/' + folder + '_concentration.png'
    plt.savefig(saveName, bbox_inches='tight', dpi=600)
    
    
# Load matlab files
def matToDf(file):
    """
    Convert .MAT file info structure into a dataframe
    """
    spMatFile = loadmat(file)

    df = pd.DataFrame.from_dict(spMatFile['info'], orient='index')
    df = df.T
    
    # Convert values to floats where possible
    for key in df.keys():
        if key != 'timestamp':
            try:
                df[key] = df[key].astype(float)
            except:
                pass
    
    return df

import scipy.io as spio
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

# MISC
def loadFolder(folder = '', plotFit = True, saveFig = False, 
               rejectTimeStart = 1, rejectTimeEnd = 2, normalise = True, wait = False):
    """
    folder: string of folder name inside ./Data containing all the MAT files
    """

    lifetime=[]; concentration=[]; standd=[]; chisq=[]; timestamps=[];
    for file in getFiles(folder):

        print('Analysing ' + file + '...', end="")
        
        # Load Data
        spMatFile = loadmat(file)
        
        ## Info
        pumpPower = spMatFile['info']['laserCurrent']
        pumpTime = spMatFile['info']['laserPulseWidth']
        sample = spMatFile['info']['sample']
        laserPulsePeriod = spMatFile['info']['laserPulsePeriod']
        timestamp = spMatFile['timestamp']
        
        try:
            concentration.append(spMatFile['info']['concentration'])
        except:
            concentration.append(np.NAN)

        # Signal
        time = spMatFile['t']
        intensity = spMatFile['buffer_a_mv_mean']    

        name = 'pumpPower' + str(pumpPower) + 'pumpTime' + str(pumpTime) + 'Chip' + sample
        
        # rejectTimeEnd = laserPulsePeriod - pumpTime - rejectTimeEnd
        
        [tau, tauErr, chi2] = getLifetime(time, intensity, sample, name, rejectTimeStart, 
                                  rejectTimeEnd, plotFit, saveFig, normalise)
        lifetime.append(tau)
        standd.append(tauErr)
        chisq.append(chi2)
        timestamps.append(timestamp)
        print('OK')
        
    print('\nDone!\n')
 
    return {'lifetime':lifetime, 'standd':standd, 'timestamps':timestamps, 'concentration':concentration}

    