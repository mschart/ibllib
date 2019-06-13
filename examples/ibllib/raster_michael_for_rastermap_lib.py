from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from oneibl.one import ONE
import ibllib.io.alf as alf

# get the data from flatiron and the current folder
one = ONE()
eid = one.search(subject='ZM_1150', date='2019-05-07', number=1)
D = one.load(eid[0], clobber=False, download_only=True)
session_path = Path(D.local_path[0]).parent

# load objects
spikes = alf.load_object(session_path, 'spikes')
clusters = alf.load_object(session_path, 'clusters')
channels = alf.load_object(session_path, 'channels')
trials = alf.load_object(session_path, '_ibl_trials')


def bin_data_neurons_x_amps(spikes,bin_width):

 '''
 This function creates a matrix which is neurons x observations

 INPUT:
 
 spikes: object being spikes = alf.load_object(session_path, 'spikes')
 bin_width: temporal bin size in sec 

 OUTPUT:

 Matrix of dimenions neurons x observations, 
 with each observation being the mean amplitude of this neuron within one time bin

 spikes is dictionary containing the output of kilosort2, 
 in particular for each spike: time, amplitude, cluster (neuron)
 '''
 
 spike_times=spikes['times']
 spike_clusters=spikes['clusters']
 spike_amps=spikes['amps']
 
 channels=np.unique(spike_clusters)
 d=[]
 
 for channel in channels:
  
  r=ma.masked_where(spike_clusters==[channel],spike_clusters)
  d.append([spike_times[ma.getmask(r)[:,0]],spike_amps[ma.getmask(r)[:,0]]])
  
 #get maximum time
 times=[[min(x[0]),max(x[0])] for x in d]
 MIN=min(np.array(times)[:,0]) 
 MAX=max(np.array(times)[:,1])

 n_bins=int((MAX-MIN)/bin_width)

 D=np.zeros((len(d),n_bins))
 
 for i in range(len(d)): 
  Counts, _ = np.histogram(d[i][0],weights=np.reshape(d[i][1],np.shape(d[i][0])),bins =n_bins, range = (MIN,MAX))
  D[i]=Counts

 return D

#D=bin_data(get_neurons_X_activity(),0.01)
# after pip install rastermpa, in bash type the following to start a GUI and load D: python -m rastermap
