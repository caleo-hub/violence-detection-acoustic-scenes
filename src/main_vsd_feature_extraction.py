from dataset_preprocessing.vsd_feature_extractor import FilmFeatureExtractor
from dataset_preprocessing.vsd_dataset_generator import VSD_DatasetGenerator


vsd_gen = VSD_DatasetGenerator(
    path="C:/Users/CSANT321/Documents/TCC/violence-detection-acoustic-scenes/Datasets/VSD_2014_December_official_release/Hollywood-dev/annotations"
)
movie_annotations = vsd_gen.optimize_annotations()

feature_extraction = FilmFeatureExtractor(
    annotations=movie_annotations,
    mat_files_path="C:/Users/CSANT321/Documents/TCC/violence-detection-acoustic-scenes/Datasets/VSD_2014_December_official_release/Hollywood-dev/features",
)

feature_extraction.process_all_films(output_file_path="src/h5_files/vsd_features.h5")

feature_extraction.create_labeled_features(
    features_h5_path="src/h5_files/vsd_features.h5",
    movie_annotations=movie_annotations,
    output_h5_path="src/h5_files/vsd_clipped_features.h5",
)
