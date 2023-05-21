import os
import h5py
import librosa
import numpy as np
import xml.etree.ElementTree as ET


class MIVIADatasetIntegrator:
    def __init__(self):
        self.annotations = []

    def read_xml_files(self, folder_path):
        self.annotations = []
        path_names = []

        # Percorre todos os arquivos no diretório especificado
        for filename in os.listdir(folder_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(folder_path, filename)

                # Faz o parsing do arquivo XML
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Percorre todos os itens no XML
                for item in root.findall(".//item"):
                    class_id = int(item.find(".//CLASS_ID").text)
                    path_name = str(item.find(".//PATHNAME").text)

                    if path_name in path_names:
                        continue

                    path_names.append(path_name)

                    # Verifica se o CLASS_ID corresponde a gunshot ou scream
                    if class_id == 3:
                        annotation = "gunshot"
                    elif class_id == 4:
                        annotation = "scream"
                    else:
                        continue

                    start_second = float(item.find(".//STARTSECOND").text)
                    end_second = float(item.find(".//ENDSECOND").text)

                    # Obtém o nome do arquivo de áudio correspondente
                    audio_file_name = os.path.splitext(filename)[0]

                    self.annotations.append(
                        (
                            audio_file_name,
                            start_second,
                            end_second,
                            annotation,
                            path_name,
                        )
                    )

    def integrate_dataset(self, audio_folder, output_file):
        audio_data = []
        annotation_data = []

        annotation_dict = {
            "scream": 1,
            "gunshot": 3,
        }

        num_audio_files = len(self.annotations)
        for i in range(1, 6):
            for j, annotation in enumerate(self.annotations):
                file_path = os.path.join(audio_folder, annotation[0] + f"_{i}.wav")

                # Carrega o áudio com a taxa de amostragem de 16000 Hz
                audio, sr = librosa.load(file_path, sr=16000)

                start_second = annotation[1]
                end_second = annotation[2]

                # Calcula os índices de início e fim do recorte de áudio
                start_index = int(start_second * sr)
                end_index = int(end_second * sr)

                # Faz o recorte do áudio
                if len(audio) < 10 * sr:
                    # Áudio menor que 10 segundos, realiza o padding com zeros
                    audio_padded = np.pad(
                        audio, (0, 10 * sr - len(audio)), mode="constant"
                    )
                    audio_data.append(audio_padded)
                else:
                    # Áudio maior que 10 segundos, considera apenas os 10 segundos centrais
                    center_index = len(audio) // 2
                    start_index = center_index - 5 * sr
                    end_index = center_index + 5 * sr
                    audio_cut = audio[start_index:end_index]
                    audio_data.append(audio_cut)

                # Armazena a anotação correspondente
                annotation_data.append(annotation_dict[annotation[3]])

                print(f"Processados: {j + 1}/{num_audio_files} áudios")

        # Salva os dados de áudio e anotação no arquivo HDF5
        with h5py.File(output_file, "a") as f:
            # Cria o grupo 'audio' e armazena os datasets com os áudios recortados
            try:
                audio_group = f.create_group("audio")
                annotation_group = f.create_group("annotations")
            except ValueError:
                audio_group = f["audio"]
                annotation_group = f["annotation"]

            for i, (audio, annotation) in enumerate(zip(audio_data, annotation_data)):
                # Nome do arquivo original de áudio

                # Nome do arquivo original seguido de um índice
                dataset_name = f"{i}"

                audio_group.create_dataset(dataset_name, data=audio)
                annotation_group.create_dataset(dataset_name, data=[annotation])

        print(f"Total de áudios processados: {num_audio_files}")
