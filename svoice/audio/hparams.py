# from tensorflow.contrib.training import HParams
from glob import glob
import os
import pickle5 as pickle

import random

def get_image_list(data_root, split):
	filelist = glob(data_root + '*')

	if split == 'train':
		filelist = filelist[:int(.95 * len(filelist))]
	else:
		filelist = filelist[int(.95 * len(filelist)):]
	
	return filelist

def get_filelist2(dataset, data_root, split):
    speakers = os.listdir(data_root)
    filelist =[]
    for i in range(len(speakers)):
        pkl_file = os.getcwd()+'/pickle_files/filenames_{}_{}_{}.pkl'.format(dataset, split,speakers[i])
        if os.path.exists(pkl_file):
            files = pickle.load(open(pkl_file, 'rb'))
        else:
            #create the pickle file
            
            files = glob('{}/{}/*/audio.wav'.format(data_root, speakers[i]))
            if split == 'train':
                    files = files[:int(.95 * len(filelist))]
            else:
                files = files[int(.95 * len(filelist)):]
            
            with open(pkl_file, 'wb') as p:
                pickle.dump(files, p, protocol=pickle.HIGHEST_PROTOCOL)
        filelist.append(files)
    return filelist

def get_filelist(dataset, data_root, split,numspeakers):
    # pkl_file = 'filenames_{}_{}.pkl'.format(dataset, split)
    # if os.path.exists(pkl_file):
    #     with open(pkl_file, 'rb') as p:
    #         return pickle.load(p)
    #else:
    #select directorires with the number of speakers 
    speakers_audio = []
    now = -1
    for i in range(numspeakers):
        pkl_file = 'filenames_{}_{}_{}.pkl'.format(dataset, split,i)
        if os.path.exists(pkl_file):
            now = i
            with open(pkl_file, 'rb') as p:
                speakers_audio.append(pickle.load(p))   
                
    speakers = os.listdir(data_root)
    #randomly select numspeakers
    speakers_choosen = random.sample(speakers,numspeakers)
    #get the filelist
    
    for i in range(numspeakers):
        #print("i: ",i,"now: ",now)
        if i >now:
            pkl_file = 'filenames_{}_{}_{}.pkl'.format(dataset, split,i)
            filelist = glob('{}/{}/*/audio.wav'.format(data_root, speakers_choosen[i]))
            #print('{}/{}/*/audio.wav'.format(data_root, speakers_choosen[i]))
            if split == 'train':
                filelist = filelist[:int(.95 * len(filelist))]
            else:
                filelist = filelist[int(.95 * len(filelist)):]
            speakers_audio.append(filelist)

            with open(pkl_file, 'wb') as p:
                pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)

    return speakers_audio



class HParams:
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.data[key]

	def set_hparam(self, key, value):
		self.data[key] = value
		
# Default hyperparameters
hparams = HParams(
    
    
    hidden_units=256,  # Alias = E
    num_layers=2,  # Alias = L
    dropout=0.05,  # Alias = D
    
    
    
    
    
    
    
    
    
	num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
    #  network
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.9,  # Rescaling value

    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
    # also consider clipping your samples to smaller chunks)
    max_mel_frames=900,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
    #  and still getting OOM errors.
    
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=False,
    
    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    n_fft_den=512,
    hop_size_den=160,
    win_size_den=400,
    
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
    # faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
    # be too big to avoid gradient explosion, 
    # not too small for fast convergence)
    normalize_for_wavenet=True,
    # whether to rescale to [0, 1] for wavenet. (better audio quality)
    clip_for_wavenet=True,
    # whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
    
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
    # levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    
    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    fmax=7600,  # To be increased/reduced depending on data.
    
    # Griffin Lim
    power=1.5,
    # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,
    # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
    ###########################################################################################################################################
    
    N=25,
    img_size=96,
    fps=25,
        
    n_gpu=3,
    batch_size=4,
    num_workers=10,
    initial_learning_rate=1e-3,
    reduced_learning_rate=None,
    nepochs=200,
    ckpt_freq=1,
    validation_interval=3,

    wav_step_size=16000,
    mel_step_size=16,
    spec_step_size=100,
    wav_step_overlap=3200
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)






