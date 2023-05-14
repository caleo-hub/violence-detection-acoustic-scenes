import h5py
import librosa
import numpy as np
import scipy.signal as signal
from tqdm import tqdm


class AudioFeatureExtractor:
    def __init__(self, filename, file_output):
        self.filename = filename
        self.features_file = file_output

    def envelope_amplitude(self, y, window_size):
        analytic_signal = signal.hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)

        sub = [0]
        for i in range(0, len(amplitude_envelope), window_size):
            y_window = y[i: i + window_size]
            sub.append(np.mean(y_window))

        return np.array(sub)

    def extract_features(self, n_fft):
        with h5py.File(self.filename, "r") as f:
            annotation_group = f["annotations"]
            audio_group = f["audio"]

            keys = list(annotation_group.keys())
            audio = [audio_group[key][:] for key in keys if len(audio_group[key][:]) > 0]
            annotations = [annotation_group[key][()] for key in keys if len(audio_group[key][:]) > 0]

        with h5py.File(self.features_file, "w") as f:
            self.features_group = f.create_group("features")
            self.annotations_group = f.create_group("annotations")

            for i, (audio_array, annotation) in enumerate(tqdm(zip(audio, annotations), total=len(audio), desc='Extracting features')):
                ae = self.envelope_amplitude(y=audio_array, window_size=n_fft).reshape((-1, 1))
                rms = librosa.feature.rms(y=audio_array, frame_length=n_fft, hop_length=n_fft)[0].reshape((-1, 1))
                zcr = librosa.feature.zero_crossing_rate(y=audio_array, frame_length=n_fft, hop_length=n_fft)[0].reshape((-1, 1))
                ber = librosa.feature.spectral_bandwidth(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1, 1))
                sc = librosa.feature.spectral_centroid(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1, 1))
                bw = librosa.feature.spectral_bandwidth(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1, 1))
                sf = librosa.onset.onset_strength_multi(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1, 1))
                mfcc = librosa.feature.mfcc(y=audio_array, sr=16000, n_mfcc=22, n_fft=n_fft, hop_length=n_fft)

                feature_array = np.concatenate([ae, rms, zcr, ber, sc, bw, sf, mfcc.T], axis=1)

                self.annotations_group.create_dataset(f"{i}", data=np.array(annotation))
                self.features_group.create_dataset(f"{i}", data=np.array(feature_array))
