import numpy as np
import sigpy as sp
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import os
from scipy.interpolate import interp1d

''' Functions for filtering and processing data '''
def get_phase(pt, ref_idx=0):
    ''' Get phase for each coil relative to ref coil '''
    pt_phase = np.empty(pt.shape)
    for i in range(pt.shape[-1]):
        pt_phase[:,i] = np.unwrap(np.angle(pt[:,i]*np.conj(pt[:,ref_idx])))
        pt_phase[:,i] *= 180/np.pi # Convert to degrees
    return pt_phase

def bpt_reshape(bpt, dims, fwd=True):
    ''' Reshape data '''
    # Reshape data to the right dimensions
    npe, ncoils, nph = dims
    if fwd: # (npe, ncoils, nph) --> (npe*nph, ncoils)
#         npe, ncoils, nph = bpt.shape
        bpt_t = np.transpose(bpt,(0,2,1)) # Put coils as the last dimension - this is so that reshaping works as intended
        bpt_r = np.reshape(bpt_t, (npe*nph,ncoils), order="F")
    else: # (npe*nph, ncoils) --> (npe, ncoils, nph)
#         bpt_t = np.transpose(bpt,(1,0)) 
        bpt_r = np.reshape(bpt, (npe,nph,ncoils), order="F")
        bpt_r = np.transpose(bpt_r, (0,2,1)) # (npe, ncoils, nph)
        
    return bpt_r

def lin_correct_all_phases(bpt_inp, corr_drift=True, demean=True):
    ''' Linear correct across phases '''
    # Try linear fit across all phases for single coil
    npe, ncoils, nph = bpt_inp.shape
    bpt_corr = np.empty((npe*nph, ncoils))
    
    # Generate vectors
    y,x = np.meshgrid(np.linspace(-1,1,nph), np.linspace(-1,1,npe))
    yy = y.flatten('F') # Drift over phases
    xx = x.flatten('F') # Drift over phase encodes
    c = np.ones((npe*nph))
    mtx = np.array([yy, xx, c]).T # Flatten fortran order; [npe*nph x 3]
    
    #  Fit per-coil
    for i in range(ncoils):
        data = np.abs(bpt_inp[:,i,:]) # size [npe x nph]
        b = data.flatten('F') # Data
        coeffs = np.linalg.inv(mtx.T @ mtx) @ mtx.T @ b
        if corr_drift: # correct for drift over phases
            bpt_corr[:,i] = b - (yy*coeffs[0] + xx*coeffs[1] + np.ones(npe*nph)*coeffs[2])
        else:
            if demean:
                bpt_corr[:,i] = b - (xx*coeffs[1] + np.ones(npe*nph)*coeffs[2])
            else:
                bpt_corr[:,i] = b - (xx*coeffs[1])
    return bpt_corr

def get_ratio(pt_loading, pt_nonloading):
    ncoils = pt_loading.shape[-1]
    pt_ratio = np.array([(np.abs(pt_nonloading[:,i]) - np.abs(pt_loading[:,i]))/np.abs(pt_nonloading[:,i])*100 for i in range(ncoils)])
    pt_ratio = np.transpose(pt_ratio, (1,0)) # Return to same dimension as PT
    return pt_ratio

def get_mean(pt):
    pt_mean = np.mean(np.abs(pt),axis=0)
    pt_idxs = np.flip(np.argsort(pt_mean))
    return pt_mean, pt_idxs

def get_max_energy(pt,tr,f_range=[0.8,1]):
    ''' Get the coil with max energy in cardiac band '''
    energy = np.empty(pt.shape[-1])
    # Get PSD
    pt_f_all = np.empty((pt.shape[0]*2, pt.shape[1]))
    for i in range(pt.shape[-1]):
        pt_f,f = zpad_1d(pt[:,i],fs=1/tr, N=None)
        pt_f_all[:,i] = np.abs(pt_f)**2
    # Normalize
    pt_f_all /= np.amax(pt_f_all)
    f_ind = np.where(np.logical_and(f >= f_range[0],f <= f_range[1]))[0]
    energy = np.sum(pt_f_all[f_ind,:], axis=0)
    max_inds = np.flip(np.argsort(energy))
    
#     for i in range(pt.shape[-1]):
#         pt_f,f = zpad_1d(pt[:,i]/np.amax(pt[:,i]),
# #                          fs=1/tr, N=None)
#         f_ind = np.where(np.logical_and(f >= f_range[0],f <= f_range[1]))
#         energy[i] = np.sum(np.abs(pt_f[f_ind])**2)
    return energy, max_inds

def get_percent_mod(pt):
    pt_mod = (pt/np.mean(pt,axis=0)-1)*100
    return pt_mod

def zpad_1d(inp, fs, N=None):
    if N is None:
        N = inp.shape[0]
    inp_zp = np.pad(inp,(N,0),'constant', constant_values=0)
    win_orig = np.hanning(inp.shape[-1])
    win_new = np.ones(inp_zp.shape[-1])
    win_new[:inp.shape[-1]//2] = win_orig[:inp.shape[-1]//2]
    win_new[-inp.shape[-1]//2:] = win_orig[-inp.shape[-1]//2:]
    inp_f_zp = sp.fft(inp_zp*win_new)
    N_zp = inp_zp.shape[0]
    f_zp = fs*np.arange(-N_zp/2,N_zp/2)*1/N_zp
    return inp_f_zp, f_zp

def get_coeffs(cutoff, fs, order=5, btype='low'):
    ''' Generate the low pass filter coefficients '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype,
                        analog=False)
    return b, a

def med_filt_c(sig, kernel_size=11):
    ''' Median filter over all coils '''
    sig_out = sig.copy()
    for c in range(sig.shape[-1]):
        sig_out[...,c] = signal.medfilt(sig[...,c], kernel_size=kernel_size)
    return sig_out

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

def np_pca(data, threshold=True, k=10):
    ''' PCA along coil dim'''
    U,S,VH = np.linalg.svd(data, full_matrices=False)
    # Back project
    if threshold:
        X = np.dot(U[...,:k],np.diag(S[:k]))
    else:
        X = np.dot(U,np.diag(S))
    return X,S

def ica(data, k=4):
    ''' ICA with sklearn fastICA '''
    ica = FastICA(n_components=k)
    ICA_comps = ica.fit_transform(data)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matr   
    return ICA_comps
    
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
    if from_front is True: # Remove from front
        phys_crop = phys[int(phys_diff//tr_phys):]
    else:
        phys_crop = phys[-int(bpt_len//tr_phys):]
    return phys_crop

# Account for DDA
def pad_bpt(bpt, npe=256, nph=30, dda=4, kind="linear"):
    ''' Interpolate BPT signal to account for dda in FIESTA scans'''
    # Create array of sampled points and points to interpolate to
    npts = (npe+dda)*nph
    interp_inds = np.arange(npts)
    del_inds = np.array([np.arange(dda) + npe*i for i in range(nph)]).flatten()
    sampled_inds = np.delete(interp_inds, del_inds)
    # Interpolate with scipy
    ncoils = bpt.shape[-1]
    bpt_interp = np.empty((npts, ncoils))
    for c in range(ncoils):
        f = interp1d(sampled_inds, bpt[...,c], kind=kind, fill_value="extrapolate")
        bpt_interp[...,c] = f(interp_inds)
    return bpt_interp

def get_physio_waveforms(inpdir, bpt_len=None,
                         tr_ppg=10e-3, tr_ecg=1e-3,
                         load_ppg=True, load_ecg=True, from_front=True, index=0):
    ''' Load ECG and PPG data based on input directory. First ECG by default '''
    phys_waveforms = [] # Order is [ecg, ppg]
    if load_ecg is True:
        ecg = load_physio(inpdir, ftype="ECG")[index,:] # First ECG
        ecg_crop = crop_physio(ecg, bpt_len, tr_phys=tr_ecg, from_front=from_front)
        phys_waveforms.append(ecg_crop)
    if load_ppg is True:
        ppg = np.squeeze(load_physio(inpdir, ftype="PPG"))
        ppg_crop = crop_physio(ppg, bpt_len, tr_phys=tr_ppg, from_front=from_front)
        phys_waveforms.append(ppg_crop)
    return phys_waveforms
    
def get_t_axis(N, delta_t):
    ''' Get time axis based on number of samples and sample spacing '''
    return np.arange(N)*delta_t
     
def sort_coils(sorting_arr):
    ''' Return sorted array from most to least '''
    return np.flip(np.argsort(sorting_arr))
    
def get_S(ref_idx, c):
    ''' Return matrix with one row of -1 '''
    S = np.zeros((c,c))
    S[ref_idx,:] = -1
    return S

def get_filtered_phase(pt, ref_idx, k=3):
    ''' Project phase onto k principal components '''
    bpt_phase = np.unwrap(np.angle(pt.T * np.conj(pt[:,ref_idx]).T).T, axis=0)
    U,S,VH = np.linalg.svd(bpt_phase, full_matrices=False)
    data_filt = bpt_phase @ VH[:,:k] @ VH[:,:k].T
    return data_filt

def get_phase_pinv(pt, k=3, pca=True, c_inds=None):
    # pt = np.squeeze(cfl.readcfl(os.path.join(inpdir,"pt_ravel")))
    N,c = pt.shape
    # Specify input coils
    if c_inds is not None:
        pt = pt[...,c_inds]
        N, c = pt.shape
    
    # Get full A matrix
    A_all = np.vstack([np.eye(c) + get_S(i,c).T for i in range(c)]) # [c^2 x c]

    # Get full data matrix
    if pca is True: # Filter with PCA first
        pt_phase_all =  np.vstack([get_filtered_phase(pt, ref_idx=i, k=k).T for i in range(c)]) # [c^2 x N]
    else:
        # pt_phase_all =  np.hstack([np.unwrap(np.angle(pt.T * np.conj(pt[:,i]).T).T) for i in range(c)]).T
        pt_phase_all =  np.hstack([np.angle(pt.T * np.conj(pt[:,i]).T).T for i in range(c)]).T
        # FOR DEBUGGING
        # print(pt_phase_all.shape)

    # Compute pseudo inverse
    A_all_pinv = np.linalg.pinv(A_all) # [c x c^2]

    # Compute phase
    pt_phase = (A_all_pinv @ np.unwrap(pt_phase_all, axis=1)).T # [N x c]
    # pt_phase = (A_all_pinv @ pt_phase_all).T # [N x c]
    
    # Convert to degrees
    # pt_phase *= 180/np.pi
    
    return pt_phase, pt