from torch.utils.data import Dataset

import os
import pandas as pd
import torchaudio

class UrbanSounDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):                # -> annotations_file = files that all annotations (strings 주석)
                                                                    # -> audio_dir = path to where we store the audio samples
        self.annotations = pd.read_csv(annotations_file)             # -> load csv pandas dataFrame
        self.audio_dir  = audio_dir

    def __len__(self):
        # -> want to return number of datasets
        return len(self.annotations)

    # len(usd)    

    def __getitem__(self,index):
        # -> getting loading wavefrom audio samples , at the same time return to label

        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path) # -> torchaudio loading the data // each OS have differ function // torchaudio.load , torchaudio.load_wav and torchaudio.save

        return signal , label

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0]) # -> find a wav file 
        return path

    def _get_audio_sample_label(self,index):        # -> take a label information
        return self.annotations.iloc[index,6]



    # what item is useful // a_list[1] -> a_list.__getitem__(1)

if __name__ == "__main__" : 

    ANNOTATIONS_FILE =  "D:/PyAud/PyTorch-Audio/3_Creating_a_custom_datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR        =  "D:/PyAud/PyTorch-Audio/3_Creating_a_custom_datasets/UrbanSound8K/audio"

    usd = UrbanSounDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]

    a = 1