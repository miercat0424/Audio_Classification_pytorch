from torch.utils.data import Dataset

import torch
import os
import pandas as pd
import torchaudio

class UrbanSounDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):                # -> annotations_file = files that all annotations (strings 주석)
                                                                    # -> audio_dir = path to where we store the audio samples
        self.annotations = pd.read_csv(annotations_file)             # -> load csv pandas dataFrame
        self.audio_dir  = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        # -> want to return number of datasets
        return len(self.annotations)

    # len(usd)    

    def __getitem__(self,index):
        # -> getting loading wavefrom audio samples , at the same time return to label

        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path) # -> torchaudio loading the data // each OS have differ function // torchaudio.load , torchaudio.load_wav and torchaudio.save
        # signal -> pytorch 2Dimesion tensor // (num_channels , sampls) -> (2,16000) -> (1,16000) // 2 = stereo
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)    # -> trying to make mono
        signal = self.transformation(signal)            # -> mel_spectrogram passing the signal
        return signal , label

    def _resample_if_necessary(self,signal,sr):         # -> already have in torchaudio but wanna configure 
        if sr != self.target_sample_rate :              # -> if the sample rate has differ than change it to all the same
            resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self,signal):            # -> if signal is stereo than change it to mono
        if signal.shape[0] > 1:                         # -> signal.shape[0] = number of channels
            signal = torch.mean(signal,dim=0,keepdim=True)
        return signal

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0]) # -> find a wav file 
        return path

    def _get_audio_sample_label(self,index):            # -> take a label information
        return self.annotations.iloc[index,6]



    # what item is useful // a_list[1] -> a_list.__getitem__(1)

if __name__ == "__main__" : 

    ANNOTATIONS_FILE =  "D:/PyAud/PyTorch-Audio/3_Creating_a_custom_datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR        =  "D:/PyAud/PyTorch-Audio/3_Creating_a_custom_datasets/UrbanSound8K/audio"
    SAMPLE_RATE      =  16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram( # -> can transform what you want 
        sample_rate=SAMPLE_RATE,
        n_fft      =1024,                           # -> frame size
        hop_length=512,                             # -> set it the half of n_fft
        n_mels=64                                   # -> 
        )


    usd = UrbanSounDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    a=1

    # -> signal shape (1 = channels , 64 = n_mels , 10 = number of frame )