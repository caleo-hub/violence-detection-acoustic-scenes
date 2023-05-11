from dataset_preprocessing.audio_dataset_integration import AudioDatasetIntegrator
from dataset_preprocessing.audio_feature_extractor import AudioFeatureExtractor

audio_dataset = AudioDatasetIntegrator()
audio_dataset.get_audio_dataset(
    path="Datasets", file_output="src/h5_files/audio_dataset.h5"
)


feature_extractor = AudioFeatureExtractor(
    filename="src/h5_files/audio_dataset.h5",
    file_output="src/h5_files/audio_features.h5",
)

feature_extractor.extract_features(n_fft=640)
