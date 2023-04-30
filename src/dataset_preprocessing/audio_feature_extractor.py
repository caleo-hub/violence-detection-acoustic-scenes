import h5py
import librosa
import numpy as np
import scipy.signal as signal


class AudioFeatureExtractor:
    def __init__(self, filename, file_output):
        self.filename = filename
        self.features_file = file_output
        
    def envelope_amplitude(self, y, window_size):
        analytic_signal = signal.hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Criando um array vazio para armazenar os valores subamostrados
        sub = [0]

        # Iterando sobre o sinal
        for i in range(0, len(amplitude_envelope), window_size):
            # Obtendo a janela atual
            y_window = y[i:i+window_size]

            # Calculando a média dos valores da janela e adicionando ao array de subamostragem
            sub.append(np.mean(y_window))

        return np.array(sub)
        
    def extract_features(self, n_fft):
        with h5py.File(self.filename, "r") as f:
            keys = list(f.keys())
            audio = []
            annotations = []
            for key in keys:
                audio_array = f[key][:]
                if len(audio_array) == 0:
                    continue
                audio.append(audio_array)
                annotations.append(f[key].attrs["annotation"])

        with h5py.File(self.features_file, "a") as f:
            try:
                features_group = f.create_group("features")
                annotations_group = f.create_group("annotations")
            except ValueError:
                features_group = f["features"]
                annotations_group = f["annotations"]
                
            for i, (audio_array, annotation) in enumerate(zip(audio, annotations)):
                
                # AE - Amplitude Envelope
                ae = self.envelope_amplitude(y=audio_array, window_size=n_fft).reshape((-1,1))

                # RMS - Root-Mean-Square
                rms = librosa.feature.rms(y=audio_array, frame_length=n_fft, hop_length=n_fft)[0].reshape((-1,1))

                # ZCR - Zero-Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y=audio_array, frame_length=n_fft, hop_length=n_fft)[0].reshape((-1,1))

                # BER - Band Energy Ratio
                ber = librosa.feature.spectral_bandwidth(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1,1))

                # SC - Spectral Centroid
                sc = librosa.feature.spectral_centroid(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1,1))

                # BW - Frequency Bandwidth
                bw = librosa.feature.spectral_bandwidth(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1,1))

                # SF - Spectral Flux
                sf = librosa.onset.onset_strength_multi(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0].reshape((-1,1))

                # MFCC - Mel-frequency cepstral coefficients
                mfcc = librosa.feature.mfcc(y=audio_array, sr=16000, n_mfcc=22, n_fft=n_fft, hop_length=n_fft)
                
                # print('ae:', ae.shape)
                # print('rms:', rms.shape)
                # print('zcr:', zcr.shape)
                # print('ber:', ber.shape)
                # print('sc:', sc.shape)
                # print('bw:', bw.shape)
                # print('sf:', sf.shape)
                # print('mfcc:', mfcc.T.shape)

                # concatena todas as features em um único array
                feature_array = np.concatenate([ae, rms, zcr, ber, sc, bw, sf, mfcc.T], axis=1)

                if i % 1000 == 0:
                    print(f"Processado {i} audios")
                            
                feature_ds = features_group.create_dataset(f"{i}",
                                                           data=np.array(feature_array))
                annotation_ds = annotations_group.create_dataset(f"{i}", data=np.array(annotation))
