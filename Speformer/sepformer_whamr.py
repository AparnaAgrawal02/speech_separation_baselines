from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import os
models = ["sepformer-libri2mix","resepformer-wsj02mix" ,"sepformer-whamr", "sepformer-wsj02mix","sepformer-Wham"]
datset_path = "/scratch/aparna/wham_noise/tt"
print(models)

store_path = "/ssd_scratch/cvit/aparna/wham_noise/sepformer/"
if not os.path.exists(store_path):
    os.makedirs(store_path)
    
for i in range(len(models)):
    name = models[i]
    if os.path.exists(store_path+name):
        continue
    os.makedirs(store_path+name, exist_ok=True)
    model = separator.from_hparams(source="speechbrain/{}".format(models[i]), savedir='pretrained_models/{}'.format(models[i]))

    for root , dirs, files in os.walk(datset_path):
        for file in files:
            print(file)
                    
            #preprocess file to smaple rate 8000
            waveform, sample_rate = torchaudio.load(os.path.join(root,file))
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)(waveform)
            torchaudio.save("temp.wav", waveform, 8000)
            if file.endswith(".wav"):
                print(file)
                if not os.path.exists(store_path+name+"/source1"):
                    os.makedirs(store_path+name+"/source1", exist_ok=True)
                if not os.path.exists(store_path+name+"/source2"):
                    os.makedirs(store_path+name+"/source2", exist_ok=True)
                est_sources = model.separate_file("temp.wav") 
                torchaudio.save(store_path+name+"/source1/"+file, est_sources[:, :, 0].detach().cpu(), 8000)
                torchaudio.save(store_path+name+"/source2/"+file, est_sources[:, :, 1].detach().cpu(), 8000)
                print("name: ", name, "file: ", file, "done")