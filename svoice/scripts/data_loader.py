from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import audio.audio_utils as audio
import audio.hparams as hparams
import random
import os
import librosa
import math
#impor F
import torch.nn.functional as F
import torchaudio
     
class DataGenerator(Dataset):

    def __init__(self, train_path1, sampling_rate, split,numspeakers,dataset_name):

        self.files = hparams.get_filelist2(dataset_name,train_path1, split) 
        
        #print("files: ",self.files)
        self.sampling_rate = sampling_rate
        self.num_speakers = numspeakers
        

    def __len__(self):
        #print("speakers: ",self.num_speakers,"files: ",len(self.files))
        #print([len(self.files[i]) for i in range(self.num_speakers)])
        return sum([len(self.files[i]) for i in range(len(self.files))])

    def __getitem__(self, index):
        segment = 1.0
        stride = 1.0
        length = int(self.sampling_rate * segment)
        stride = int(self.sampling_rate * stride)
        while(1):
            aduio_files = []
            
            #stfts of source files
            waves = []
            stft = None
            indexs = []
            for i in range(self.num_speakers):
                c=0
                while(1):
                    c+=1
                    index = random.randint(0, len(self.files) - 1)
                    if index in indexs:
                        continue
                    indexs.append(index)
                    if len(self.files[index]) == 0:
                        #print("0 audios in this speaker")
                        continue
                    index2 = random.randint(0, len(self.files[index]) - 1)
                    #stft = self.process_audio(self.files[index][index2])
                    #wave = audio.load_wav(self.files[index][index2], self.sampling_rate)
                    arr, org_sr = torchaudio.load(self.files[index][index2], num_frames=length)
                    wave = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=self.sampling_rate)[0]
                    wave = F.pad(wave, (0, length - wave.shape[0]), "constant").data.numpy()
                    print("wave:",wave.shape)
                    if wave is not None:
                        #print(c)
                        break
                    
                aduio_files.append(self.files[index][index2])
                
                waves.append(wave)
            #print("stfts:",stfts[0].shape)

            #mix the audio files
            mix_audio = self.mix_audio(waves)
            mix_audio = F.pad(mix_audio, (0, length - mix_audio.shape[0]), "constant")
            print("mix_audio:",mix_audio.shape)
            temp_mix_file = "temp_mix{}.wav".format(index)
            #save
            #librosa.output.write_wav(temp_mix_file, mix_audio, self.sampling_rate)

            #fname = self.files[index]

            #stft = self.process_audio(temp_mix_file)
            #print("stft:",stft)

            # if  stft is None :
            #     #print("stft is None",stft)
            #     continue

            #inp_mel = torch.FloatTensor(np.array(mel)).unsqueeze(1)
            #inp_stft = torch.FloatTensor(np.array(stft))
            #gt_stft = torch.FloatTensor(np.array(y))
            #stfts = [torch.FloatTensor(np.array(stft)) for stft in stfts]
            waves = [torch.FloatTensor(np.array(wave)) for wave in waves]
            print("mix_audio1:",mix_audio.shape[0])
            #os.remove(temp_mix_file)
            return mix_audio ,torch.LongTensor([mix_audio.shape[0]]), torch.stack(waves)


    def process_audio(self, file):

        # Load the gt wav file
        try:
            gt_wav = audio.load_wav(file, self.sampling_rate)                   # m
        except:
            return None

        # # Get the random file from VGGSound to mix with the ground truth file
        # random_file = random.choice(self.random_files)

        # # Load the random wav file
        # try:
        #     random_wav = audio.load_wav(random_file, self.sampling_rate)        # n
        # except:
        #     return None, None, None

        # # Mix the noisy wav file with the clea min([len(self.files[i]) for i in range(self.num_speakers)])n GT file
        # try:
        #     idx = random.randint(0, len(random_wav) - len(gt_wav) - 1)
        #     random_wav = random_wav[idx:idx + len(gt_wav)]
        #     snrs = [0, 5, 10]
        #     target_snr = random.choice(snrs)
        #     noisy_wav = self.add_noise(gt_wav, random_wav, target_snr)
        # except:
        #     return None, None, None

        # Extract the corresponding audio segments of 1 second
        start_idx, gt_seg_wav = self.crop_audio_window(gt_wav)
        
        if start_idx is None or gt_seg_wav is None :
            return None


        # -----------------------------------STFT min([len(self.files[i]) for i in range(self.num_speakers)])s--------------------------------------------- #
        # Get the STFT, normalize and concatenate the mag and phase of GT and noisy wavs
        gt_spec = self.get_spec(gt_seg_wav)                                     # Tx514

        #noisy_spec = self.get_spec(noisy_seg_wav)                               # Tx514 
        # ------------------------------------------------------------------------------------- #


        # -----------------------------------Melspecs------------------------------------------ #                          
        #noisy_mels = self.get_segmented_mels(start_idx, noisy_wav)              # Tx80x16
        #if noisy_mels is None:
        #    return None, None, None
        # ------------------------------------------------------------------------------------- #
        
        # Input to the lipsync student model: Noisy melspectrogram
        #inp_mel = np.array(noisy_mels)                                          # Tx80x16

        # Input to the denoising model: Noisy linear spectrogram
        #inp_stft = np.array(noisy_spec)                                         # Tx514

        # GT to the denoising model: Clean linear spectrogram
        gt_stft = np.array(gt_spec)                                             # Tx514
        #print("gt_stft:",gt_stft.shape)
        
        return gt_stft


    def crop_audio_window(self, gt_wav):

        if gt_wav.shape[0] - hparams.hparams.wav_step_size <= 1280: 
            return None, None

        # Get 1 second random segment from the wav
        start_idx = np.random.randint(low=1280, high=gt_wav.shape[0] - hparams.hparams.wav_step_size)
        end_idx = start_idx + hparams.hparams.wav_step_size
        gt_seg_wav = gt_wav[start_idx : end_idx]
        
        if len(gt_seg_wav) != hparams.hparams.wav_step_size: 
            return None, None

        # noisy_seg_wav = noisy_wav[start_idx : end_idx]
        # if len(noisy_seg_wav) != hparams.hparams.wav_step_size: 
        #     return None, None, None

        # # Data augmentation
        # aug_steps = np.random.randint(low=0, high=3200)
        # aug_start_idx = np.random.randint(low=0, high=hparams.hparams.wav_step_size - aug_steps)
        # aug_end_idx = aug_start_idx+aug_steps

        # aug_types = ['zero_speech', 'reduce_speech', 'increase_noise']
        # aug = random.choice(aug_types)

        # if aug == 'zero_speech':    
        #     noisy_seg_wav[aug_start_idx:aug_end_idx] = 0.0
            
        # elif aug == 'reduce_speech':
        #     noisy_seg_wav[aug_start_idx:aug_end_idx] = 0.1*gt_seg_wav[aug_start_idx:aug_end_idx]

        # elif aug == 'increase_noise':
        #     random_seg_wav = random_wav[start_idx : end_idx]
        #     noisy_seg_wav[aug_start_idx:aug_end_idx] = gt_seg_wav[aug_start_idx:aug_end_idx] + (2*random_seg_wav[aug_start_idx:aug_end_idx])

        return start_idx, gt_seg_wav


    def crop_mels(self, start_idx, noisy_wav):
        
        end_idx = start_idx + 3200

        # Get the segmented wav (0.2 second)
        noisy_seg_wav = noisy_wav[start_idx : end_idx]
        if len(noisy_seg_wav) != 3200: 
            return None
        
        # Compute the melspectrogram using librosa
        spec = audio.melspectrogram(noisy_seg_wav, hparams.hparams).T              # 16x80
        spec = spec[:-1] 

        return spec


    def get_segmented_mels(self, start_idx, noisy_wav):

        mels = []
        if start_idx - 1280 < 0: 
            return None

        # Get the overlapping continuous segments of noisy mels
        for i in range(start_idx, start_idx + hparams.hparams.wav_step_size, 640): 
            m = self.crop_mels(i - 1280, noisy_wav)                             # Hard-coded to get 0.2sec segments (5 frames)
            if m is None or m.shape[0] != hparams.hparams.mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)                                                 # Tx80x16

        return mels


    def get_spec(self, wav):

        # Compute STFT using librosa
        stft = librosa.stft(y=wav, n_fft=hparams.hparams.n_fft_den, \
               hop_length=hparams.hparams.hop_size_den, win_length=hparams.hparams.win_size_den).T
        stft = stft[:-1]                                                        # Tx257

        # Decompose into magnitude and phase representations
        mag = np.abs(stft)
        mag = audio.db_from_amp(mag)
        phase = audio.angle(stft)

        # Normalize the magnitude and phase representations
        norm_mag = audio.normalize_mag(mag)
        norm_phase = audio.normalize_phase(phase)
            
        # Concatenate the magnitude and phase representations
        spec = np.concatenate((norm_mag, norm_phase), axis=1)               # Tx514
        
        return spec

    def add_noise(self, gt_wav, random_wav, desired_snr):

        samples = len(gt_wav)

        signal_power = np.sum(np.square(np.abs(gt_wav)))/samples
        noise_power = np.sum(np.square(np.abs(random_wav)))/samples

        k = (signal_power/(noise_power+1e-8)) * (10**(-desired_snr/10))

        scaled_random_wav = np.sqrt(k)*random_wav

        noisy_wav = gt_wav + scaled_random_wav

        return noisy_wav
    
    def mix_audio(self,waves):
        # load audio files with librosa and mix them
        mixed_wav = np.zeros_like(waves[0])
        for w in waves:
            mixed_wav += w
        #convert to torch tensor
        mixed_wav = torch.from_numpy(mixed_wav)
        return mixed_wav
        

        
       
def load_data( train_path,num_workers,num_spearkers, batch_size=4, split='train', sampling_rate=16000, shuffle=False):
    
    dataset = DataGenerator(train_path, sampling_rate, split,num_spearkers,dataset_name="Lrs2")

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return data_loader