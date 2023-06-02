import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio
import os
import librosa
# input can be a URL or a local path
#input = 'https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav'

name = "speech_mossformer_separation_temporal_8k"
datset_path = "/scratch/aparna/wham_noise/tt"
store_path = "/ssd_scratch/cvit/aparna/wham_noise/mossformer/"
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer_separation_temporal_8k')


for root , dirs, files in os.walk(datset_path):
        for file in files:
            print(file)
                    
            #load using sf to convert to 8000
            waveform, sample_rate = librosa.load(os.path.join(root,file), sr=8000)
            sf.write("temp.wav",waveform, 8000)
            if file.endswith(".wav"):
                print(file)
                if not os.path.exists(store_path+name+"/source1"):
                    os.makedirs(store_path+name+"/source1", exist_ok=True)
                if not os.path.exists(store_path+name+"/source2"):
                    os.makedirs(store_path+name+"/source2", exist_ok=True)
                result = separation("temp.wav")
                for i, signal in enumerate(result['output_pcm_list']):
                    save_file = store_path+name+"/source"+str(i+1)+"/"+file
                    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
    
        
                print("name: ", name, "file: ", file, "done")