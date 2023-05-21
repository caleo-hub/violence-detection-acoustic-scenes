from dataset_preprocessing.audio_feature_extractor import AudioFeatureExtractor


def main():
    datasets_path = [
        ("GunshotForensic", "src/h5_files/GunshotForensic_dataset.h5"),
        ("SESA", "src/h5_files/SESA_dataset.h5"),
        ("HEAR_Train", "src/h5_files/HEAR_Train_dataset.h5"),
        ("HEAR_Test", "src/h5_files/HEAR_Test_dataset.h5"),
        ("Audioset", "src/h5_files/Audioset_dataset.h5"),
    ]

    for dataset_name, dataset_path in datasets_path:
        output_path = f"src/h5_files/{dataset_name}_feature.h5"
        print(f"Processing {dataset_name} dataset")
        feature_extractor = AudioFeatureExtractor(dataset_path, output_path)
        feature_extractor.extract_features(n_fft=640)
        print(f"{dataset_name} dataset processed and annotated")


if __name__ == "__main__":
    main()
