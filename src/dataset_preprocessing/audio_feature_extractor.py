import h5py
import librosa
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
import math


class AudioFeatureExtractor:
    def __init__(self, filename, file_output):
        self.filename = filename
        self.features_file = file_output

    def envelope_amplitude(self, y, window_size):
        analytic_signal = signal.hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)

        sub = []
        for i in range(0, len(amplitude_envelope), window_size):
            y_window = y[i : i + window_size]
            sub.append(np.mean(y_window))

        return np.array(sub)

    def normalize_feature(self, feature):
        return 255 * (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

    def calculate_split_frequency_bin(
        self, split_frequency, sample_rate, num_frequency_bins
    ):
        frequency_range = sample_rate / 2
        frequency_delta_per_bin = frequency_range / num_frequency_bins
        split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
        return int(split_frequency_bin)

    def band_energy_ratio(self, spectrogram, split_frequency, sample_rate):
        split_frequency_bin = self.calculate_split_frequency_bin(
            split_frequency, sample_rate, len(spectrogram[0])
        )
        band_energy_ratio = []

        power_spectrogram = np.abs(spectrogram) ** 2
        power_spectrogram = power_spectrogram.T

        for frame in power_spectrogram:
            sum_power_low_frequencies = frame[:split_frequency_bin].sum()
            sum_power_high_frequencies = frame[split_frequency_bin:].sum()
            band_energy_ratio_current_frame = (
                sum_power_low_frequencies / sum_power_high_frequencies
            )
            band_energy_ratio.append(band_energy_ratio_current_frame)

        return np.array(band_energy_ratio)

    def extract_features(self, n_fft, split_frequency=8000, sr=16000):
        with h5py.File(self.filename, "r") as f:
            annotation_group = f["annotations"]
            audio_group = f["audio"]

            keys = list(annotation_group.keys())
            audio = [
                audio_group[key][:] for key in keys if len(audio_group[key][:]) > 0
            ]
            annotations = [
                annotation_group[key][()]
                for key in keys
                if len(audio_group[key][:]) > 0
            ]

        with h5py.File(self.features_file, "w") as f:
            self.features_group = f.create_group("features")
            self.annotations_group = f.create_group("annotations")

            for i, (audio_array, annotation) in enumerate(
                tqdm(
                    zip(audio, annotations),
                    total=len(audio),
                    desc="Extracting features",
                )
            ):
                spec = librosa.stft(audio_array, n_fft=n_fft, hop_length=n_fft)
                ber = self.normalize_feature(
                    self.band_energy_ratio(
                        spectrogram=spec,
                        split_frequency=split_frequency,
                        sample_rate=sr,
                    )
                ).reshape((-1, 1))

                ae = self.normalize_feature(
                    self.envelope_amplitude(y=audio_array, window_size=n_fft)
                ).reshape((-1, 1))
                rms = self.normalize_feature(
                    librosa.feature.rms(
                        y=audio_array, frame_length=n_fft, hop_length=n_fft
                    )[0]
                ).reshape((-1, 1))
                zcr = self.normalize_feature(
                    librosa.feature.zero_crossing_rate(
                        y=audio_array, frame_length=n_fft, hop_length=n_fft
                    )[0]
                ).reshape((-1, 1))

                sc = self.normalize_feature(
                    librosa.feature.spectral_centroid(
                        y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft
                    )[0]
                ).reshape((-1, 1))
                bw = self.normalize_feature(
                    librosa.feature.spectral_bandwidth(
                        y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft
                    )[0]
                ).reshape((-1, 1))
                sf = self.normalize_feature(
                    librosa.onset.onset_strength_multi(
                        y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft
                    )[0]
                ).reshape((-1, 1))
                mfcc = self.normalize_feature(
                    librosa.feature.mfcc(
                        y=audio_array,
                        sr=16000,
                        n_mfcc=22,
                        n_fft=n_fft,
                        hop_length=n_fft,
                    )
                )

                feature_list = [ae, rms, zcr, ber, sc, bw, sf, mfcc.T]
                for feature_num, feature in enumerate(feature_list):
                    size = feature.shape
                    tiled_features = np.tile(feature, (251 // size[0] + 1, 1))
                    feature_list[feature_num] = tiled_features[:251, :]

                feature_array = np.concatenate(feature_list, axis=1)

                self.annotations_group.create_dataset(f"{i}", data=np.array(annotation))
                self.features_group.create_dataset(f"{i}", data=np.array(feature_array))
