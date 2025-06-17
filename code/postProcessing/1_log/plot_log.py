'''
    plot log data
'''

import os, sys, matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, LogLocator


# %%

class Parameter:
    casename, runname = 'RBC_PTV_1E7_07', 'run2' 

# %%


def main(): 
    params = Parameter()
    params.log_path = '../../../data/'+params.casename+'/output/'+params.runname+'/logs/training_log.txt'
    params.save_path = '../../../data/'+params.casename+'/output/'+params.runname+'/logs/'
    
    # plot format
    Fontsize, Linewidth, Tickwidth, Pad = 15, 1.3, 2, 10
    
    # load training history
    log = np.loadtxt(params.log_path,delimiter=',',skiprows=1)
    x = np.linspace(1,len(log),len(log))
    
    # plot losses
    minv, maxv = np.min(log[:,3:9]), np.max(log[:,3:9])
    log_min, log_max = np.floor(np.log10(minv)), np.ceil(np.log10(maxv))
    ticksy = [10**x for x in range(int(log_min), int(log_max) + 1)]
    ticksx = [10**x for x in range(0, int(np.ceil(np.log10(len(log))))+1)]
    fig, ax = plt.subplots(figsize=(7,5),dpi=300)
    ax.set_xlim(ticksx[0],ticksx[-1]), ax.set_ylim(ticksy[0],ticksy[-1])
    ax.set_xticks(ticksx, ticksx, fontsize=Fontsize-2,fontweight='bold')
    ax.set_yticks(ticksy, ticksy, fontsize=Fontsize-2,fontweight='bold')
    ax.tick_params(axis='both', which='major', width=Tickwidth), ax.tick_params(axis='both', which='minor', width=Tickwidth)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2)), ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('epoch', fontsize=Fontsize, labelpad=Pad, fontweight='bold'), ax.set_ylabel('loss', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    ax.plot(x,log[:,3], c='purple', linewidth=Linewidth, label=r'$\mathcal{L}_{tot}$')
    ax.plot(x,log[:,4], c='green', linewidth=Linewidth, label=r'$\mathcal{L}_{data}$')
    ax.plot(x,log[:,6], c='blue', linewidth=Linewidth, label=r'$\mathcal{L}_{NS}$')
    ax.plot(x,log[:,7], c='orange', linewidth=Linewidth, label=r'$\mathcal{L}_{EE}$')
    ax.plot(x,log[:,8], c='red', linewidth=Linewidth, label=r'$\mathcal{L}_{div}$')
    ax.plot(x,log[:,5], c='gray', linewidth=Linewidth, label=r'$\mathcal{L}_{bound}$')
    ax.plot(x,log[:,2], c='black', linewidth=2, label='lr')
    ax.grid(), ax.semilogx(), ax.semilogy()
    plt.tight_layout(), plt.legend()
    plt.savefig(params.save_path+'loss.jpg')
    plt.close('all')
    
    # plot mae   
    minv, maxv = np.min(log[:,9:14]), np.max(log[:,9:14])
    log_min, log_max = np.floor(np.log10(minv)), np.ceil(np.log10(maxv))
    ticksy = [10**x for x in range(int(log_min), int(log_max) + 1)]
    fig, ax = plt.subplots(figsize=(7,5),dpi=300)
    ax.set_xlim(ticksx[0],ticksx[-1]), ax.set_ylim(ticksy[0],ticksy[-1])
    ax.set_xticks(ticksx, ticksx, fontsize=Fontsize-2,fontweight='bold')
    ax.set_yticks(ticksy, ticksy, fontsize=Fontsize-2,fontweight='bold')
    ax.tick_params(axis='both', which='major', width=Tickwidth), ax.tick_params(axis='both', which='minor', width=Tickwidth)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2)), ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('epoch', fontsize=Fontsize, labelpad=Pad, fontweight='bold'), ax.set_ylabel('MAE', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    ax.plot(x,log[:,9], c='red', linewidth=Linewidth, label='u')
    ax.plot(x,log[:,10], c='green', linewidth=Linewidth, label='v')
    ax.plot(x,log[:,11], c='blue', linewidth=Linewidth, label='w')
    ax.plot(x,log[:,12], c='orange', linewidth=Linewidth, label='T')
    ax.plot(x,log[:,13], c='purple', linewidth=Linewidth, label='p')
    ax.grid(), ax.semilogx(), ax.semilogy()
    plt.tight_layout(), plt.legend()
    plt.savefig(params.save_path+'MAE.jpg')
    plt.close('all')
    
    # plot pcc    
    fig, ax = plt.subplots(figsize=(7,5),dpi=300)
    ax.set_xlim(ticksx[0],ticksx[-1]), ax.set_ylim(0,105)
    ax.set_xticks(ticksx, ticksx, fontsize=Fontsize-2,fontweight='bold')
    ax.set_yticks([0,20,40,60,80,100],[0,20,40,60,80,100], fontsize=Fontsize-2,fontweight='bold')
    ax.tick_params(axis='both', which='major', width=Tickwidth), ax.tick_params(axis='both', which='minor', width=Tickwidth)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2)), ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('epoch', fontsize=Fontsize, labelpad=Pad, fontweight='bold'), ax.set_ylabel('PCC [%]', fontsize=Fontsize, labelpad=Pad, fontweight='bold')
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    ax.plot(x,log[:,14]*100, c='red', linewidth=Linewidth, label='u')
    ax.plot(x,log[:,15]*100, c='green', linewidth=Linewidth, label='v')
    ax.plot(x,log[:,16]*100, c='blue', linewidth=Linewidth, label='w')
    ax.plot(x,log[:,17]*100, c='orange', linewidth=Linewidth, label='T')
    ax.plot(x,log[:,18]*100, c='purple', linewidth=Linewidth, label='p')
    ax.grid(), ax.semilogx()
    plt.tight_layout(), plt.legend()
    plt.savefig(params.save_path+'PCC.jpg')
    plt.close('all')
if __name__ == "__main__":
    main()