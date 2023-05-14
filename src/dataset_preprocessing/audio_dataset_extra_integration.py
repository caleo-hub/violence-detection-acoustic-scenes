import os
import librosa
import numpy as np
import pandas as pd
import h5py


class AudioExtraDatasetIntegrator:
    def get_audio_dataset(self, path, file_output):
        audio_info_df = self.organize_dataframe(self.extract_audio_info(folder=path))
        annotations = self.annotations_loader(audio_info_df)
        self.audio_loader(audio_info_df, annotations, h5_filepath=file_output)

    def annotations_loader(self, audio_info_df):
        annotation_dict = {
            "(nothing)": 0,
            "scream": 1,
            "Violência Física": 2,
            "gunshot": 3,
            "explosions": 4,
        }

        annotation_list = (
            audio_info_df["Anotação"].apply(lambda x: annotation_dict[x]).tolist()
        )
        
        annotation_list = np.array(annotation_list, np.int16)
        return annotation_list

    def audio_loader(self, audio_info_df, annotation, h5_filepath):
        sample_rate = 16_000
        duration_seconds = 10
        with h5py.File(h5_filepath, "w") as f:
            audio_group = f.create_group(name="audio")
            annotation_group = f.create_group(name="annotation")
            for index, row in audio_info_df.iterrows():
                try:
                    audio, sr = librosa.load(row["Filepath"], sr=sample_rate)
                    
                    # Se o áudio for maior que 10 segundos, pegue os 10 segundos centrais
                    if len(audio) / sr > duration_seconds:
                        start_sample = int((len(audio) / 2) - (duration_seconds * sr / 2))
                        end_sample = start_sample + duration_seconds * sr
                        audio = audio[start_sample:end_sample]

                    new_len = sr * duration_seconds
                    padded_audio = librosa.util.pad_center(audio, size=new_len)

                    audio_group.create_dataset(str(index), data=padded_audio)
                    annotation_group.create_dataset(
                        name=str(index), data=annotation[index]
                    )

                    if index % 10 == 0:
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
            if "gunshot" in row["Filepath"]:
                df.at[i, "Classe"] = "gunshot"
                df.at[i, "Anotação"] = "gunshot"
            elif "explosion" in row["Filepath"]:
                df.at[i, "Classe"] = "explosions"
                df.at[i, "Anotação"] = "explosions"
            else:
                df.at[i, "Classe"] = ""
                df.at[i, "Anotação"] = ""
            
            df.at[i, "Início"] = 0
            df.at[i, "Final"] = df.at[i, "Duração"]

        # Filtrar linhas onde a classe não está vazia
        df = df[df['Classe'] != ""]
        df = df[df['Anotação'] != ""]
        
        df = df.reset_index()
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
