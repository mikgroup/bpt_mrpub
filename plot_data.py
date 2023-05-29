import numpy as np
import matplotlib.pyplot as plt
import processing as proc

def plot_data(data_mat, c_mat, color_dict, labels,
              figsize=(10,10), shift=-1,
              tr=8.7e-3,
              xlim=np.array([0,10]), dashed_line=True):
    ''' Plot stack of data in one plot ''' 
    # Get percent modulation
    t = proc.get_t_axis(bpt.shape[1], delta_t=tr)
    
    # Calculate time indices
    xlim_n = (xlim*1/tr).astype(int)
        
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(data_mat.shape[-1]):
        # Plot
        sig_norm = proc.normalize(data_mat[xlim_n[0]:xlim_n[1],i])
        color = color_dict[c_mat[i]]
        ax.plot(t[xlim_n[0]:xlim_n[1]], sig_norm + i*shift, color=color)
        
        # Label percent mod
        offset = 0.05
        max_value = np.max(sig_norm) + i*shift

        ax.text(t[xlim_n[1]], max_value + offset,
                "{:.2f}%".format(labels[i]),
                ha='center',
                color=color)
        
        # Label coil
        ax.text(t[xlim_n[0]], max_value + offset,
                "Coil {}".format(c_mat[i]),
                ha='center',
                color=color)
        
        # Label axes
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_yticks([])
        
    if dashed_line is True:
        # Add optional dashed line
        offset = 4.5
        ax.axhline(len(c_mat)*shift/2 + offset, linestyle='--', color='black')
        
    return ax
    
def plot_bpt_pt(bpt, cmat, title="Raw BPT vs PT Magnitude"):
    ''' Plot BPT vs PT on the same plot '''
    c_mat_r = c_mat.ravel()
    color_dict = make_color_dict(np.unique(c_mat_r))

    # Get percent mod
    pt_mod = proc.get_percent_mod(np.abs(bpt[0,...]))
    bpt_mod = proc.get_percent_mod(np.abs(bpt[1,...]))

    # Stack data into matrix
    data_mat = np.hstack((pt_mod[...,c_mat[0,:]], bpt_mod[...,c_mat[1,:]]))
    mean_percent_mod = np.mean(np.abs(data_mat), axis=0)

    ax = plot_data(data_mat, c_mat_r, color_dict, labels=mean_percent_mod, shift=-8, dashed_line=True)
    ax.set_title(title)

    
def plot_bpt_accel(bpt, accel_d, ecg, ppg, figsize=(10,10), shift=-8, c=[30,24], title="BPT vs peripherals"):
    ''' Plot BPT and PT vs accelerometer and peripherals '''
    t = proc.get_t_axis(bpt.shape[1], delta_t=tr)
    t_ecg = proc.get_t_axis(ecg.shape[0], delta_t=1e-3)
    t_ppg = proc.get_t_axis(ppg.shape[0], delta_t=10e-3)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:purple", "tab:green", "tab:gray", "tab:red", "tab:blue"]
    labels = ["PT coil {}".format(c[0]), "BPT coil {}".format(c[1]), "Accel-y", "PPG", "ECG"]

    # PT, BPT
    ax.plot(t, proc.normalize(np.abs(bpt[0,:,c[0]])), color=colors[0])
    ax.plot(t, proc.normalize(np.abs(bpt[1,:,c[1]])) + shift, color=colors[1])

    # Accelerometer
    ax.plot(t, proc.normalize(accel_d[:,1]) + 2*shift, color=colors[2])

    # Peripherals
    ax.plot(t_ppg, proc.normalize(ppg) + 3*shift, color=colors[3])
    ax.plot(t_ecg, -1*proc.normalize(ecg) + 4*shift, color=colors[4])

    # Set xlimits
    ax.set_xlim([0,10])
    ax.set_yticks([])

    # Plot labels
    for i in range(len(labels)):
        offset = 2
        # Label coil
        ax.text(t[0]-0.5, i*shift + offset,
                labels[i],
                ha='center',
                color=colors[i])
        
    # Title
    ax.set_title(title)
        
def plot_dbcg(bpt, accel_d, tr=8.7e-3, t_end=10, title="BPT dBCG comparison"):
    # Plot Figure 3 - Accelerometer vs physio
    figsize=(10,10)
    bpt_inp = proc.normalize_c(np.abs(bpt[1,...]))
    accel_inp = proc.normalize_c(accel_d)
    bpt_d = proc.get_bpt_d(accel_inp, bpt_inp)
    bpt_filt = proc.filter_c(bpt_d, cutoff=15, tr=tr)

    # Labels and colors
    labels = np.array(list(zip(["BPT-dBCG-{}".format(axis) for axis in ["x","y","z"]],["dBCG-{}".format(axis) for axis in ["x","y","z"]]))).flatten()
    colors = ["darkcyan", "darkkhaki", "purple", "tab:gray", "maroon","black"]

    # Compute correlations
    corr = np.round(np.array([np.corrcoef(bpt_filt[:,i], accel_inp[:,i])[0,1] for i in range(3)]),2)

    t = proc.get_t_axis(bpt.shape[1], delta_t=tr)
    fig, ax = plt.subplots(figsize=figsize)

    # Shift each set vertically
    shifts = np.arange(3)*-15
    offsets = np.arange(3)*-15 + np.ones(3)*-5
    all_shifts = np.array(list(zip(shifts,offsets))).flatten()
    data = np.vstack([( proc.normalize(bpt_filt[...,i]),  proc.normalize(accel_inp[...,i])) for i in range(3)]).T
    t_end = 10 # s

    for i in range(len(labels)):
        shift = all_shifts[i]

        # Plot data with shift
        ax.plot(t, data[:,i] + shift, label=labels[i], color=colors[i])

        # Label BPT, dBCG etc
        max_value = np.max(proc.normalize(data[:,i]))
        ax.text(t[0]-0.5, max_value + shift, labels[i], ha='center', color=colors[i])

        # Correlation value
        if i % 2 == 0:
            ax.text(t_end-0.5, max_value + shift, "Correlation = {}".format(corr[i//2]), ha='center', color=colors[i])
    ax.set_yticks([])
    ax.set_xlim([0,t_end])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

def make_color_dict(array):
    ''' Make a color dictionary to map numbers to colors '''
    colors = np.array(list(mcolors.TABLEAU_COLORS.keys()))

    # Predefine dictionary of size N
    N = len(array)
    color_inds = np.arange(N)
    color_dict = dict(zip(array,np.array(colors)))
    return color_dict