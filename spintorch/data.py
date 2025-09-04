# Generate Data
# Generate Continuous Souce Signal with Gaussian Noise
import torch
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import pickle
import seaborn as sns
sns.set_context('paper')


def zero(input,omega):
   #print(omega)
   if omega==0:
      return 0
   else:
      return input


def DataGenerate(a,b,dt,Bt,fbase,f_list,change_list,noise):
   'a,b are starting and ending time'
   x=np.arange(a,b,dt)
   y=list()
   omega_list = 2*np.pi*dt*fbase*np.array(f_list) #list of frequencies in radians per sample
   phi = 0; # phase accumulator
   count = int(0)
   # increment phase based on current frequency
   for j in range(len(omega_list)):
      for i in range(count,int(change_list[j]*len(x))):
         phi = phi + omega_list[j]
         c = Bt*np.sin(phi) # sine of current phase
         c = zero(c,omega_list[j])
         y.append(c)
         count+=1
   #Add Gaussian White Noise centered at 0
   if noise[0] == True:
      N_phi = np.random.normal(loc=0,scale=noise[1]*np.std(y),size=len(x))
      y += N_phi
   return x,y



def fourierTrans(dt,timesteps,y):
   from scipy.fft import fft, fftfreq
   SAMPLE_RATE = 1/dt
   DURATION = timesteps*dt
   # Number of samples in normalized_tone
   N = int(SAMPLE_RATE * DURATION)
   yf = fft(y)
   xf = fftfreq(N, 1 / SAMPLE_RATE)
   return np.abs(xf), np.abs(yf)


def data_plot(f_change_list,timesteps,dt,Bt,fbase,change_list,noise,labels,basedir):
   #----------ORIGINAL SIGNAL PLOT--------------#
   rcParams['figure.figsize'] = 12,4*len(f_change_list)
   fig, ax = plt.subplots(len(f_change_list),1,sharex=True,sharey=True)
   ax = ax.flatten()

   inputs_list = []
   outputs_list = []
   for i in range(len(f_change_list)):
      sns.lineplot(x=x, y=y, ax=ax[i], label=f'f={f_change_list[i]}GHz')
      ax[i].legend(title=f'signal{i}, label{labels[i]}',loc='upper right')
      ax[i].set(xlabel='Timesteps', ylabel='Magnitude')
      inputs_list.append(y)
      outputs_list.append(labels[i])
   plt.savefig(f'{basedir}/source_signal.png',dpi=300)

   #-----------FOURIER TANSFORM PLOT------------#
   rcParams['figure.figsize'] = 12,4*len(f_change_list)
   fig, ax = plt.subplots(len(f_change_list),1,sharex=True,sharey=True)
   ax = ax.flatten()
   for i in range(len(f_change_list)):
      x,y=DataGenerate(0, timesteps*dt,dt,Bt,fbase,f_change_list[i],change_list,noise)
      xf,yf = fourierTrans(dt,timesteps,y)
      sns.lineplot(x=xf, y=yf[:len(xf)], ax=ax[i], label=f'f={f_change_list[i]}GHz')
      #ax[i].plot(xf, yf,label=f'f={f_change_list[i]}GHz')
      ax[i].legend(title=f'signal{i} FFT, label{labels[i]}',loc='upper right')
      ax[i].set(xlabel='Frequency/Hz', ylabel='Magnitude')
   plt.savefig(f'{basedir}/source_FOURIER.png',dpi=300)


def generate_ar_signal_fm(n_samples, ar_order, ar_coeffs, vary_range, noise_std=0):
    signal = np.zeros(n_samples)
    
    # Generate initial random value
    signal[0] = np.random.randn()
    
    # Calculate the base angular frequency range for 4GHz
    base_frequency = 4e9  # 4GHz
    sampling_rate = 10e9  # Sampling rate in Hz (e.g., 10GHz)
    omega_base_min = 2 * np.pi * (base_frequency - vary_range) / sampling_rate  # Min base angular frequency
    omega_base_max = 2 * np.pi * (base_frequency + vary_range) / sampling_rate  # Max base angular frequency
    
    for t in range(1, n_samples):
        # Calculate the current value based on previous values and AR coefficients
        omega_base = np.random.uniform(omega_base_min, omega_base_max)
        for i in range(min(t, ar_order)):
            signal[t] += ar_coeffs[i] * np.sin(omega_base * (t - i - 1))
        
        # Add noise
        signal[t] += noise_std * np.random.randn()
    
    return signal   