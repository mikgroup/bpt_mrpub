import numpy as np
import sigpy as sp
from scipy import signal
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks
import scipy.integrate as integ
import os
import cfl

def normalize(sig, var=True):
    ''' Subtract mean and divide by std for 1D signal '''
    if var:
        return (sig - np.mean(sig))/np.std(sig)
    else:
        return (sig - np.mean(sig, axis=0))

def normalize_c(sig, var=True):
    ''' Whiten each coil data '''
    sig_out = sig.copy()
    for c in range(sig_out.shape[-1]):
        sig_out[...,c] = normalize(sig[...,c], var=var)
    return sig_out

def filter_sig(sig, cutoff, fs, order=6, btype='low'):
    ''' Filter the signal sig with desired cutoff in Hz and sampling freq fs in Hz '''
    # Get the filter coefficients so we can check its frequency response.
    b, a = get_coeffs(cutoff, fs, order, btype)
    # Filter
    sig_filt = signal.filtfilt(b, a, sig, padlen=50)
    return sig_filt

def get_coeffs(cutoff, fs, order=5, btype='low'):
    ''' Generate the low pass filter coefficients '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype,
                        analog=False)
    return b, a

def filter_c(bpt, cutoff=1, tr=4.4e-3):
    ''' Low pass or bandpass filter over all coils '''
    bpt_filt = np.empty(bpt.shape)
    # Check filter type  - NOTE: may cause bugs if this fails
    if type(cutoff) in [int, np.int64, float, np.float64]:
        btype = 'lowpass'
    else:
        btype = 'bandpass'
          
    # Filter per coil
    for c in range(bpt.shape[-1]):
        bpt_filt[...,c] = filter_sig(bpt[...,c],
                                     cutoff=cutoff, # Cutoff and fs in Hz
                                     fs=1/(tr), order=6, btype=btype)
    return bpt_filt

def get_percent_mod(pt):
    ''' Compute percent modulation relative to the mean '''
    pt_mod = (pt/np.mean(pt,axis=0)-1)*100
    return pt_mod

def load_physio(inpdir, ftype="PPG"):
    ''' Load physio waveform from text file '''
    # Check for text file in the input directory that starts with appropriate name
    physio_fnames = [f for f in os.listdir(inpdir) if f.startswith(ftype)]
    physio = []
    for i in range(len(physio_fnames)):
        physio.append(np.loadtxt(os.path.join(inpdir,physio_fnames[i]),
                    comments="#", delimiter=",", unpack=False))
    return np.array(physio)

def crop_physio(phys, bpt_len, tr_phys=1e-3, from_front=True):
    ''' Crop first ~30s of physio waveform '''
    phys_len = phys.shape[0]*tr_phys
    phys_diff = phys_len - bpt_len # seconds
    phys_crop = phys[int(phys_diff//tr_phys):]
    return phys_crop

def get_physio_waveforms(inpdir, bpt_len=None,
                         tr_ppg=10e-3, tr_ecg=1e-3,
                         load_ppg=True, load_ecg=True, index=0):
    ''' Load ECG and PPG data based on input directory. First ECG by default '''
    phys_waveforms = [] # Order is [ecg, ppg]
    if load_ecg is True:
        ecg = load_physio(inpdir, ftype="ECG")[index,:] # First ECG
        ecg_crop = crop_physio(ecg, bpt_len, tr_phys=tr_ecg)
        phys_waveforms.append(ecg_crop)
    if load_ppg is True:
        ppg = np.squeeze(load_physio(inpdir, ftype="PPG"))
        ppg_crop = crop_physio(ppg, bpt_len, tr_phys=tr_ppg)
        phys_waveforms.append(ppg_crop)
    return phys_waveforms

def get_t_axis(N, delta_t):
    ''' Get time axis based on number of samples and sample spacing '''
    return np.arange(N)*delta_t


def get_bpt_d(accel_d, bpt_inp):
    ''' Find coefficients to linearly combine BPT to match displacement'''
    bpt_d = np.empty(accel_d.shape)
    for i in range(accel_d.shape[1]):
        accel_inp = normalize(accel_d[:,i])
        opt_vals = lsq_linear(bpt_inp, accel_inp)
        bpt_d[:,i] = lin_comb(opt_vals.x, bpt_inp)
    return bpt_d

# Try least squares fit to calculate coeffs of x, y and z
def lin_comb(x, accel_d):
    return np.sum(x[i] * accel_d[:,i] for i in range(accel_d.shape[1]))

def get_bpt(ksp, threshold=0.05):
    ''' Extract BPT from kspace data of size [nro, npe, nframes, ncoils]'''
    # Take IFFT along readout direction
    ksp_f = sp.ifft(ksp, axes=(0,)) 
    nro, npe, nframes, ncoils = ksp_f.shape

    # Find peaks by taking root sum square over dimensions other than readout
    # Peaks are extracted if they are greater than the threshold
    ksp_f_rss = sp.rss(ksp_f, axes=(1,2,3))
    ksp_f_rss /= np.amax(ksp_f_rss)
    peaks, _ = find_peaks(ksp_f_rss, threshold=threshold)

    # Extract BPT at those locations
    bpt = ksp_f[peaks,...] # First dim is the number of BPTs

    # Reshape the BPT to be of size [nbpts, npe*nframes, ncoils]
    bpt_r = np.reshape(bpt, (peaks.shape[0], npe*nframes, ncoils), order="F")

    return bpt_r

def get_accel_data(inpdir, fname=None):
    ''' Load accelerometer data from file '''
    # Load fname as file that starts with 'data'
    if fname is None:
        fname = [f for f in os.listdir(inpdir) if f.startswith('data')][0]
    data = np.loadtxt(os.path.join(inpdir,fname))
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    accel = np.vstack([x,y,z]).T
    return accel[1:,:] # Return in the same shape as BPT

def dbl_int(accel, tr=8.7e-3, cutoff=1, get_v=False):
    ''' Double integrate acceleration -> displacement '''
    # Filter out fluctuations in accelerometer signal
    accel_filt = filter_sig(accel, cutoff=cutoff, fs=1/tr, order=6, btype='high')
    accel_v = integ.cumtrapz(accel_filt, dx=tr, initial=0)
    accel_d = integ.cumtrapz(normalize(accel_v, var=False), dx=tr, initial=0)
    if get_v is True: # Get velocity
        return accel_d, accel_v
    else:
        return accel_d

def get_accel_d(accel, tr=8.7e-3, cutoff=3, get_v=False):
    ''' Get integrated acceleration -> displacement for all axes '''
    accel_d = np.empty((accel.shape[0],3))
    if get_v is True:
        # Optionally get velocity
        accel_v = np.empty((accel.shape[0],3))
        for i in range(accel_d.shape[-1]):
            d, v = dbl_int(accel[:,i], tr=tr, cutoff=cutoff, get_v=True)
            accel_d[:,i] = d
            accel_v[:,i] = v
        return accel_d, accel_v
    else:
        # Get displacement
        for i in range(accel_d.shape[-1]):
            accel_d[:,i] = dbl_int(accel[:,i], tr=tr, cutoff=cutoff, get_v=False)
        return accel_d

def load_data(inpdir, ecg_index=1, cutoff=4):
    ''' Load accelerometer and physio data '''
    # Define parameters
    tr = 8.7e-3 # s
    
    # Load BPT
    ksp = cfl.readcfl(os.path.join(inpdir, "ksp")) # [N,N,nframes,ncoils]
    bpt = get_bpt(ksp)
    
    # Load accelerometer data and integrate to displacement
    accel = get_accel_data(inpdir)
    accel_d = get_accel_d(accel, tr=tr, cutoff=cutoff)
    
    # Load peripherals
    [ecg, ppg] = get_physio_waveforms(inpdir,
                                      bpt_len=bpt.shape[1]*tr,
                                      index=ecg_index)
    
    return bpt, ecg, ppg, accel_d