import os
import h5py
import librosa
from tqdm import tqdm


class AudioDatasetIntegrator:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path

    def process_audio_files(self):
        with h5py.File(self.output_path, "w") as f:
            audio_group = f.create_group("audio")

            for i, (dirpath, _, filenames) in enumerate(
                tqdm(os.walk(self.dataset_path), desc="Walking through directories")
            ):
                for filename in filenames:
                    if filename.endswith(".wav") or filename.endswith(".mp3"):
                        try:
                            full_path = os.path.join(dirpath, filename)
                            audio, sr = librosa.load(full_path, sr=16000)
                            if (
                                len(audio) > 160000
                            ):  # if audio is longer than 10 seconds
                                start = (
                                    len(audio) // 2 - 80000
                                )  # take 10 second slice in the middle
                                audio = audio[start : start + 160000]

                            dataset = audio_group.create_dataset(
                                f"{filename}_{i}", data=audio
                            )
                            dataset.attrs["path"] = full_path

                        except Exception as e:
                            print(f"Error processing {filename}: {e}")

    def annotate_audio_files(self):
        annotation_map = {
            "nao_violencia": 0,
            "scream": 1,
            "violencia": 2,
            "slap": 2,
            "gunshot": 3,
            "explosion": 4,
        }

        with h5py.File(self.output_path, "r+") as f:
            audio_group = f["audio"]
            annotation_group = f.create_group("annotations")

            for name, dataset in tqdm(
                audio_group.items(), desc="Annotating audio files"
            ):
                path = dataset.attrs["path"]
                lower_path = path.lower()

                for keyword, annot_value in annotation_map.items():
                    if "nao_violencia" in lower_path:
                        annotation = 0
                        break
                    if keyword in lower_path:
                        annotation = annot_value
                        break
                else:
                    annotation = 0

                annotation_dataset = annotation_group.create_dataset(
                    name, data=annotation
                )
                annotation_dataset.attrs["path"] = path
