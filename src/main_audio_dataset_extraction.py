from dataset_preprocessing.audio_dataset_integration import AudioDatasetIntegrator

def main():
    datasets_path = [
        ('HEAR_Train', 'Datasets/HEAR Dataset/AUDIO/media/tiago/ESTUDO/DataSets/HEAR/__BASE_HEAR_V2/Audio/Train'),
        ('HEAR_Test', 'Datasets/HEAR Dataset/AUDIO/media/tiago/ESTUDO/DataSets/HEAR/__BASE_HEAR_V2/Audio/Test'),
        ('GunshotForensic', 'Datasets/Gunshot Audio Forensic Dataset'),
        ('SESA', 'Datasets/SESA')
    ]

    for dataset_name, dataset_path in datasets_path:
        output_path = f'src/h5_files_new/{dataset_name}_dataset.h5'
        print(f"Processing {dataset_name} dataset")
        integrator = AudioDatasetIntegrator(dataset_path, output_path)
        integrator.process_audio_files()
        integrator.annotate_audio_files()
        print(f"{dataset_name} dataset processed and annotated")

if __name__ == "__main__":
    main()