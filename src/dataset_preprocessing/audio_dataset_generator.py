from dataset_integration import AudioDatasetIntegrator

audio_dataset = AudioDatasetIntegrator()
dataset = audio_dataset.get_audio_dataset(path='Datasets',
                                          file_output='src/h5_files/audios_with_annotation.h5')