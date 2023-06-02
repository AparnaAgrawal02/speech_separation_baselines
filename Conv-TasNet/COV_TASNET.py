from asteroid.models import ConvTasNet
import os
import torchaudio
import torch
import subprocess
#models = ["mpariente/ConvTasNet_WHAM!_sepclean","JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"]

name = "ConvTasNet"
store_path = "/ssd_scratch/cvit/aparna/REAL_M/ConvTasNet/"
datset_path = "/scratch/aparna/REAL-M-v0.1.0/audio_files_converted_8000Hz"
for root , dirs, files in os.walk(datset_path):
        for file in files:
           
                if file.endswith(".wav"):
                    print(file)
                        #preprocess file to smaple rate 8000
                    waveform, sample_rate = torchaudio.load(os.path.join(root,file))
                    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)(waveform)
                    #load wav
                    #wav,s = torchaudio.load(os.path.join(root,file))
                    #convert to (batch_size, num_channels, num_samples)
                    #print(wav.shape)
                    #input = wav.unsqueeze(0)
                    subprocess.call(["python", "Conv_TasNet_Pytorch/Separation_wav.py", "-mix_scp", os.path.join(root,file), "-yaml", "Conv_TasNet_Pytorch/options/train/train.yml", "-model", "Conv_TasNet_Pytorch/checkpoint/best.pt", "-gpuid", "0,1,2,3,4,5,6,7", "-save_path", store_path+name+"/"])
                    
                    print("name: ", name, "file: ", file, "done")
            
                