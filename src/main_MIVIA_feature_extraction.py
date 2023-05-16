from dataset_preprocessing.MIVIA_dataset_integration import MIVIADatasetIntegrator
from dataset_preprocessing.audio_feature_extractor import AudioFeatureExtractor


integrator = MIVIADatasetIntegrator()
integrator.read_xml_files('Datasets/MIVIA_DB4_dist/training')

integrator.integrate_dataset('Datasets/MIVIA_DB4_dist/training/sounds',
                             'src/h5_files/MIVIA_train_dataset.h5')


feature_extractor = AudioFeatureExtractor(
    filename="src/h5_files/MIVIA_train_dataset.h5",
    file_output="src/h5_files/MIVIA_train_features.h5",
)

feature_extractor.extract_features(n_fft=640)

integrator.read_xml_files('Datasets/MIVIA_DB4_dist/testing')

integrator.integrate_dataset('Datasets/MIVIA_DB4_dist/testing/sounds',
                             'src/h5_files/MIVIA_test_dataset.h5')


feature_extractor = AudioFeatureExtractor(
    filename="src/h5_files/MIVIA_test_dataset.h5",
    file_output="src/h5_files/MIVIA_test_features.h5",
)

feature_extractor.extract_features(n_fft=640)