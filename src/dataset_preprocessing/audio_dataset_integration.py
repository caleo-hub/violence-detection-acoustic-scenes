import os
import librosa
import numpy as np
import pandas as pd
import h5py


class AudioDatasetIntegrator:
    def get_audio_dataset(self, path, file_output):
        audio_info_df = self.organize_dataframe(self.extract_audio_info(folder=path))
        annotations = self.annotations_loader(audio_info_df)
        self.audio_loader(audio_info_df, annotations, h5_filepath=file_output)

    def annotations_loader(self, audio_info_df):
        annotation_dict = {
            "(nothing)": 0,
            "scream": 1,
            "Violência Física": 2,
            "gunshot_forensic": 3,
            "explosions": 4,
        }

        annotation_list = []
        annotation_list = (
            audio_info_df["Anotação"].apply(lambda x: annotation_dict[x]).tolist()
        )

        annotation_list = np.array(annotation_list, np.int16)

        return annotation_list

    def audio_loader(self, audio_info_df, annotation, h5_filepath):
        sample_rate = 16_000
        duration_seconds = 10

        # Cria um arquivo h5 para armazenar os áudios
        with h5py.File(h5_filepath, "w") as f:
            audio_group = f.create_group(name="audio")
            annotation_group = f.create_group(name="annotation")
            for index, row in audio_info_df.iterrows():
                try:
                    audio, sr = librosa.load(row["Filepath"], sr=sample_rate)
                    new_len = sr * duration_seconds
                    padded_audio = librosa.util.pad_center(audio, size=new_len)

                    audio_group.create_dataset(str(index), data=padded_audio)
                    annotation_group.create_dataset(
                        name=str(index), data=annotation[index]
                    )

                    if index % 1000 == 0:
                        print(f"Processado {index} audios")
                except Exception as e:
                    print(f"Erro ao carregar o arquivo {row['Filepath']}: {str(e)}")

    def extract_audio_info(self, folder):
        data = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    file_path = os.path.join(dirpath, filename)
                    duration = librosa.get_duration(path=file_path)
                    sample_rate = librosa.get_samplerate(file_path)
                    data.append([duration, file_path, sample_rate])
        df = pd.DataFrame(data, columns=["Duração", "Filepath", "Amostragem"])
        return df

    def organize_dataframe(self, df):
        df["Classe"] = ""
        df["Anotação"] = ""
        df["Tag"] = ""

        df["Início"] = ""
        df["Final"] = ""

        for i, row in df.iterrows():
            if "HEAR Dataset" in row["Filepath"] and "NAO_VIOLENCIA" in row["Filepath"]:
                df.at[i, "Classe"] = "Violência Física"
                df.at[i, "Anotação"] = "(nothing)"
            elif "HEAR Dataset" in row["Filepath"] and "VIOLENCIA" in row["Filepath"]:
                df.at[i, "Classe"] = "Violência Física"
                df.at[i, "Anotação"] = "Violência Física"
            elif "Gunshot Audio Forensic Dataset" in row["Filepath"]:
                df.at[i, "Classe"] = "gunshots"
                df.at[i, "Anotação"] = "gunshot_forensic"
                df.at[i, "Tag"] = os.path.basename(
                    os.path.dirname(row["Filepath"])
                ).replace("_Samsung", "")
            df.at[i, "Início"] = 0
            df.at[i, "Final"] = df.at[i, "Duração"]
        return df[
            [
                "Classe",
                "Duração",
                "Anotação",
                "Tag",
                "Filepath",
                "Início",
                "Final",
                "Amostragem",
            ]
        ]
