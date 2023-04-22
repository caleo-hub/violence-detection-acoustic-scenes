from dataset_preprocessing.audio_feature_extractor import AudioFeatureExtractor

audio_extractor = AudioFeatureExtractor(filename='src/h5_files/audios_with_annotation.h5',
                                        file_output='src/h5_files/audio_features.h5')

audio_extractor.extract_features(n_fft=640)