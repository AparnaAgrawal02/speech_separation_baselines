from asteroid.models import ConvTasNet
import os
import torchaudio
import torch
models = ["mpariente/ConvTasNet_WHAM!_sepclean","JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"]
for i in range(len(models)):
    name = models[i]

    model = ConvTasNet.from_pretrained(models[i])
    model.eval()
    store_path = "/ssd_scratch/cvit/aparna/wham_noise/ConvTasNet/"
    datset_path = "/scratch/aparna/wham_noise/tt"
    for root , dirs, files in os.walk(datset_path):
            for file in files:
                try:
                    if file.endswith(".wav"):
                        print(file)
                         #preprocess file to smaple rate 8000
                        waveform, sample_rate = torchaudio.load(os.path.join(root,file))
                        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)(waveform)
                        if not os.path.exists(store_path+name+"/source1"):
                            os.makedirs(store_path+name+"/source1", exist_ok=True)
                        if not os.path.exists(store_path+name+"/source2"):
                            os.makedirs(store_path+name+"/source2", exist_ok=True)
                        #load wav
                        #wav,s = torchaudio.load(os.path.join(root,file))
                        #convert to (batch_size, num_channels, num_samples)
                        #print(wav.shape)
                        #input = wav.unsqueeze(0)
                        with torch.no_grad():
                            est_sources = model(waveform.unsqueeze(0))
                        print(est_sources)
                        print(est_sources.shape)
                        torchaudio.save(store_path+name+"/source1/"+file, est_sources[:, :, 0].detach().cpu(), 8000)
                        torchaudio.save(store_path+name+"/source2/"+file, est_sources[:, :, 1].detach().cpu(), 8000)
                        print("name: ", name, "file: ", file, "done")
                except:
                    print("error")
                    continue
                    