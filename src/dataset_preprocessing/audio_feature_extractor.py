import h5py
import librosa
import numpy as np
from scipy.interpolate import interp1d



class AudioFeatureExtractor:
    def __init__(self, filename, file_output):
        self.filename = filename
        self.features_file = file_output
        
    def envelope_amplitude(self, signal, sr, window_size):
        # Define o passo de deslizamento da janela para fazer subamostragem de fator 2
        hop_length = window_size
        
        # Divide o sinal em janelas sobrepostas com tamanho window_size e passo hop_length
        signal_frames = librosa.util.frame(signal, frame_length=window_size, hop_length=hop_length)
        
        # Aplica a janela de Hann a cada janela
        window = librosa.filters.hann(window_size)
        signal_frames_windowed = signal_frames * window[:, np.newaxis]
        
        # Calcula o valor absoluto do sinal suavizado em cada janela
        signal_abs = np.abs(signal_frames_windowed)
        
        # Calcula os picos do sinal em cada janela
        peaks = np.argmax(signal_abs, axis=0)
        
        # Cria um vetor de tempo com a mesma dimensão do sinal
        time = np.arange(0, len(signal))/sr
        
        # Interpola os pontos dos picos
        interp_fn = interp1d(time[peaks], signal_abs[peaks, range(len(peaks))], kind='cubic', fill_value='extrapolate')
        
        # Retorna o envelope de amplitude interpolado
        return interp_fn(time)
        
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
                ae = self.envelope_amplitude(signal=audio_array,sr=16000, window_size=n_fft)

                # RMS - Root-Mean-Square
                rms = librosa.feature.rms(y=audio_array, frame_length=n_fft, hop_length=n_fft)[0]

                # ZCR - Zero-Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y=audio_array, frame_length=n_fft, hop_length=n_fft)[0]

                # BER - Band Energy Ratio
                ber = librosa.feature.spectral_bandwidth(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0]

                # SC - Spectral Centroid
                sc = librosa.feature.spectral_centroid(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0]

                # BW - Frequency Bandwidth
                bw = librosa.feature.spectral_bandwidth(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0]

                # SF - Spectral Flux
                sf = librosa.onset.onset_strength_multi(y=audio_array, sr=16000, n_fft=n_fft, hop_length=n_fft)[0]

                # MFCC - Mel-frequency cepstral coefficients
                mfcc = librosa.feature.mfcc(y=audio_array, sr=16000, n_mfcc=22, n_fft=n_fft, hop_length=n_fft)
                
                print('ae:', ae.shape)
                print('rms:', rms.shape)
                print('zcr:', zcr.shape)
                print('ber:', ber.shape)
                print('sc:', sc.shape)
                print('bw:', bw.shape)
                print('sf:', sf.shape)
                print('mfcc:', mfcc.T.shape)

                # concatena todas as features em um único array
                feature_array = np.concatenate([ae, rms, zcr, ber, sc, bw, sf, mfcc.T])

                if i % 1000 == 0:
                    print(f"Processado {i} audios")
                            
                feature_ds = features_group.create_dataset(f"{i}", data=np.array(feature_array))
                annotation_ds = annotations_group.create_dataset(f"{i}", data=np.array(annotation))
