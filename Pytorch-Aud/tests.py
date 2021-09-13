import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch

from torchaudio.transforms import MelSpectrogram

n_fft = 40000
win_len = None
hop_len = 512
n_mels = 128
sample_rate = 22050

path = '/Users/cmu/Desktop/AI/Pytorch-Aud/audio.wav'

waveform, sample_rate = librosa.load(path, sr=sample_rate)
waveform = torch.Tensor(waveform)

torchaudio_melspec = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_len,
    hop_length=hop_len,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
)(waveform)

librosa_melspec = librosa.feature.melspectrogram(
    waveform.numpy(),
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_len,
    win_length=win_len,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
    norm='slaney',
    htk=True,
)

mse = ((torchaudio_melspec - librosa_melspec) ** 2).mean()

print(f'MSE:\t{mse}')

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('Mel Spectrogram')

axs[0].set_title('torchaudio')
axs[0].set_ylabel('Log')
axs[0].set_xlabel('time')
# axs[0].imshow(librosa.amplitude_to_db(torchaudio_melspec), aspect='auto')
Data = librosa.amplitude_to_db(torchaudio_melspec)
img1 = librosa.display.specshow(Data,y_axis="log",x_axis="time",sr=sample_rate,ax=axs[0])

axs[1].set_title('librosa')
axs[1].set_ylabel('Log')
axs[1].set_xlabel('time')
img2 = librosa.display.specshow(Data,y_axis="log",x_axis="time",sr=sample_rate,ax=axs[1])
# axs[1].imshow(librosa.power_to_db(torchaudio_melspec), aspect='auto')
# axs[1].imshow(librosa.power_to_db(librosa_melspec), aspect='auto')
fig.savefig("spec.png")
plt.show()

