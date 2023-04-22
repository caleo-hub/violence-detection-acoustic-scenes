from dataset_preprocessing.dataset_integration import AudioDatasetIntegrator
import h5py
          
audio_dataset = AudioDatasetIntegrator()
data_dict = audio_dataset.get_audio_dataset(path = 'C:/Users/CSANT321/Documents/TCC/violence-detection-acoustic-scenes/Datasets')


filename = '/dataset_preprocessing/audio_dataset.h5'
with h5py.File(filename, "w") as f:
    for key in data_dict.keys():
        f.create_dataset(key, data=data_dict[key])

