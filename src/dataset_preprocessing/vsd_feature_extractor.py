import os
import numpy as np
import h5py
import librosa

class FilmFeatureExtractor:
    def __init__(self, annotations, mat_files_path):
        self.annotations = annotations
        self.mat_files_path = mat_files_path

    def process_mat_file(self, mat_file_path):
        with h5py.File(mat_file_path, 'r') as mat_data:
            ae = np.array(mat_data['AE'])
            rms = np.array(mat_data['RMS'])
            zcr = np.array(mat_data['ZCR'])
            ber = np.array(mat_data['BER'])
            sc = np.array(mat_data['SC'])
            bw = np.array(mat_data['BW'])
            sf = np.array(mat_data['SF'])
            mfcc = np.array(mat_data['MFCC'])

            feature_array = np.concatenate([ae, rms, zcr, ber, sc, bw, sf, mfcc], axis=1)
            
            return feature_array

    def process_all_films(self, output_file_path):
        with h5py.File(output_file_path, 'w') as hf:
            for film_name in self.annotations:
                print('Processando:', film_name)
                mat_file_path = os.path.join(self.mat_files_path, f"{film_name}_auditory.mat")
                if os.path.exists(mat_file_path):
                    feature_array = self.process_mat_file(mat_file_path)
                    hf.create_dataset(film_name, data=feature_array)
                else:
                    print(f"Arquivo .mat não encontrado para o filme {film_name}.")
    
    def create_labeled_features(self, features_h5_path, movie_annotations, output_h5_path):
        annotation_dict = {
            '(nothing)': 0,
            'scream': 1,
            'Violência Física': 2,
            'gunshot': 3,
            'explosion': 4,
            'multiple_actions': 5
        }

        target_length = 251

        with h5py.File(features_h5_path, 'r') as features_h5, h5py.File(output_h5_path, 'w') as output_h5:
            annotations_group = output_h5.create_group('annotations')
            features_group = output_h5.create_group('features')

            for movie, movie_features in features_h5.items():
                for idx, (start_time, end_time, annotation, _, _) in enumerate(movie_annotations[movie]):
                    annotation = annotation_dict.get(annotation, annotation_dict['(nothing)'])
                    start_sample = np.floor(start_time * 25).astype(int)
                    end_sample = np.ceil(end_time * 25).astype(int)
                    total_samples = end_sample - start_sample

                    num_slices = int(np.ceil(total_samples / target_length))
                    for i in range(num_slices):
                        slice_start = start_sample + i * target_length
                        slice_end = min(slice_start + target_length, end_sample)
                        sliced_features = movie_features[slice_start:slice_end, :].astype(np.float32)
                        padded_features = librosa.util.pad_center(sliced_features, size=target_length, axis=0)
                        dataset_name = f"{movie}_{idx}_{i}"
                        features_group.create_dataset(dataset_name, data=padded_features)
                        annotations_group.create_dataset(dataset_name, data=np.array([annotation], dtype=np.int8)) 
                        print(dataset_name)