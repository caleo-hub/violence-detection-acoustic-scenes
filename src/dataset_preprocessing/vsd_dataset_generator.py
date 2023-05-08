import os
import glob
import matplotlib.pyplot as plt


class VSD_DatasetGenerator:
    
    def __init__(self, path):
        self.path = path
        self.class_names = ['explosions', 'gunshots', 'screams']

        self.txt_files = self.get_filenames()
        
    def get_filenames(self):
        # usa o módulo glob para encontrar arquivos com extensão .txt
        
        filenames = glob.glob(os.path.join(self.path, "*.txt"))
        # filtra apenas os arquivos que contenham algum dos itens de self.class_names em seu nome
        filenames_filtered = list(filter(lambda x: any(class_name in x for class_name in self.class_names), filenames))
        
        return filenames_filtered
              
    def get_movie_names(self):
        # inicializa o set de nomes de arquivos
        movie_names = set()

        # adiciona o nome de cada arquivo encontrado ao set,
        # sem a informação após '_'
        for txt_file in self.txt_files:
            name = os.path.basename(txt_file).split("_")[0]
            movie_names.add(name)

        return movie_names
    
    def get_movie_original_annotations(self):
        
        movie_names = self.get_movie_names()
        movie_annotations = {movie:[] for movie in movie_names}
        movies_filename = {movie:[] for movie in movie_names}
        
        for movie_name in movie_names:
            movies_filename[movie_name] = list(filter(lambda x: movie_name in x, self.txt_files))

        for movie, filenames in movies_filename.items():
            for filename in filenames:
                file_path = os.path.join(self.path, filename)
                with open(file_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            duration = round(end_time - start_time, 3)
                            if len(parts) >= 3:
                                annotation = parts[2]
                                tag = parts[3] if len(parts) == 4 else ''
                                
                        movie_annotations[movie].append((start_time,
                                                        end_time,
                                                        annotation,
                                                        tag,
                                                        duration))
        return movie_annotations
            

    def plot_film_annotations(self, film_name, annotations=None):
        if annotations is None:
            annotations = self.get_movie_original_annotations()
        
        # Cria dicionários separados para cada tipo de annotation
        colors = {
            'explosion': 'red',
            'gunshot': 'green',
            'scream': 'blue',
            'scream_effort': 'black',
            'multiple_actions': 'orange',
            '(nothing)': 'grey',
            'others': 'purple',
        }
        
        annotation_list = annotations[film_name]
        
        # Cria uma figura e um eixo
        fig, ax = plt.subplots(figsize=(20, 3))
        
        for start_time, end_time, annotation, _, duration in annotation_list:
            # Calcula a largura e posição do retângulo
            width = round(duration, 2)
            x = round(start_time, 2)
            y = 0  # posicao vertical inicial
            
            # Adiciona o retângulo à lista correspondente ao tipo de annotation
            rect_color = colors.get(annotation, colors['others'])
            rect = plt.Rectangle((x, y), width, 1, color=rect_color, alpha=0.5)
            ax.add_patch(rect)
        
        # Adiciona as legendas
        custom_lines = [plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()]
        ax.legend(custom_lines, colors.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.set_xlim(0, annotation_list[-1][1])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Annotations')
        ax.set_title(film_name)
        
        # Remove os ticks do eixo y
        ax.tick_params(axis='y', which='both', length=0)
        plt.title(film_name)

        plt.show()

    def remove_nothing_annotations(self, movie_annotations):
        no_nothing_annotations = {}
        for movie, annotations in movie_annotations.items():
            no_nothing_annotations[movie] = [annotation for annotation in annotations if annotation[2] != '(nothing)']
        return no_nothing_annotations
        
    def remove_inner_annotations(self, annotations):
        temp_annotations = []
        for i, current_annotation in enumerate(annotations):
            is_inner = False
            for j, other_annotation in enumerate(annotations):
                if i != j and current_annotation[0] >= other_annotation[0] and current_annotation[1] <= other_annotation[1]:
                    is_inner = True
                    break
            if not is_inner:
                temp_annotations.append(current_annotation)
        return temp_annotations
    
    def find_intersections(self, temp_annotations):
        updated_annotations = []
        added = set()

        for i, current_annotation in enumerate(temp_annotations):
            if i in added:
                continue

            has_intersection = False
            for j, other_annotation in enumerate(temp_annotations):
                if i != j and current_annotation[1] > other_annotation[0] and current_annotation[0] < other_annotation[1]:
                    has_intersection = True
                    start_time = min(current_annotation[0], other_annotation[0])
                    end_time = max(current_annotation[1], other_annotation[1])
                    duration = round(end_time - start_time, 3)
                    updated_annotations.append((start_time, end_time, current_annotation[2], current_annotation[3], duration))
                    added.add(j)
                    break

            if not has_intersection:
                updated_annotations.append(current_annotation)
                added.add(i)

        return updated_annotations
    
    def add_nothing_annotations(self, annotations):
        complete_annotations = annotations.copy()
        timeline = sorted(annotations, key=lambda x: x[0])
        
        complete_annotations.append((0.0, timeline[0][0], '(nothing)', '', timeline[0][0]))
        for i in range(len(timeline) - 1):
            current_annotation = timeline[i]
            next_annotation = timeline[i + 1]
            
            if current_annotation[1] < next_annotation[0]:
                start_time = current_annotation[1]
                end_time = next_annotation[0]
                duration = end_time - start_time
                complete_annotations.append((start_time, end_time, '(nothing)', '', duration))
        
        return sorted(complete_annotations, key=lambda x: x[0])
    
    def optimize_annotations(self):
        original_annotations = self.get_movie_original_annotations()
        no_nothing_annotations = self.remove_nothing_annotations(original_annotations)
        
        optimized_annotations = {}
        for movie, annotations in no_nothing_annotations.items():
            temp_annotations = self.remove_inner_annotations(annotations)
            updated_annotations = self.find_intersections(temp_annotations)
            complete_annotations = self.add_nothing_annotations(updated_annotations)
            optimized_annotations[movie] = complete_annotations
        
        return optimized_annotations
            

if __name__ == '__main__':
    vsd_gen = VSD_DatasetGenerator(path='C:/Users/CSANT321/Documents/TCC/violence-detection-acoustic-scenes/Datasets/VSD_2014_December_official_release/Hollywood-dev/annotations')
    movie_annotations = vsd_gen.get_movie_original_annotations()
    new_annotations = vsd_gen.optimize_annotations()
    vsd_gen.plot_film_annotations('Armageddon', new_annotations)
