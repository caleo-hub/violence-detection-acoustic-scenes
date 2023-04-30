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
            

def plot_film_annotations(film_name, annotations):
    # Cria dicionários separados para cada tipo de annotation
    explosions = {'heights': [], 'color': 'red', 'label': 'Explosions'}
    gunshots = {'heights': [], 'color': 'green', 'label': 'Gunshots'}
    screams = {'heights': [], 'color': 'blue', 'label': 'Screams'}
    scream_effort = {'heights': [], 'color': 'black', 'label': 'Scream Effort'}
    multiple_actions = {'heights': [], 'color': 'orange', 'label': 'Multiple Actions'}
    nothing = {'heights': [], 'color': 'grey', 'label': 'Nothing'}
    others = {'heights': [], 'color': 'purple', 'label': 'Others'}
    
    annotation_list = annotations[film_name]
    
    # Cria um dicionário com as funções de tratamento de cada tipo de annotation
    annotation_handlers = {
        'explosion': lambda x, y, w: explosions['heights'].append((x, y, w)),
        'gunshot': lambda x, y, w: gunshots['heights'].append((x, y + 1, w)),
        'scream': lambda x, y, w: screams['heights'].append((x, y + 2, w)),
        'scream_effort': lambda x, y, w: scream_effort['heights'].append((x, y + 3, w)),
        'multiple_actions': lambda x, y, w: multiple_actions['heights'].append((x, y + 4, w)),
        '(nothing)': lambda x, y, w: nothing['heights'].append((x, y + 5, w)),
        'others': lambda x, y, w: others['heights'].append((x, y + 6, w))
    }
    
    for start_time, end_time, annotation, _, duration in annotation_list:
        # Calcula a largura e posição do retângulo
        width = duration
        x = start_time
        y = 0  # posicao vertical inicial
        # Adiciona o retângulo à lista correspondente ao tipo de annotation
        annotation_handlers.get(annotation, annotation_handlers['others'])(x, y, width)
        
    # Cria uma figura e um eixo
    fig, ax = plt.subplots(figsize=(20, 3))
    
    # Plota retângulos para cada tipo de annotation
    heights = [explosions, gunshots, screams, scream_effort, multiple_actions, nothing, others]
    
    for i, h in enumerate(heights):
        if len(h['heights']) == 0:
            continue 
        xs, ys, widths = zip(*h['heights'])
        rects = ax.bar(xs, 1, widths, bottom=ys, color=h['color'], alpha=0.5, label=h['label'])
        for rect in rects:
            rect.set_linewidth(0)
    
    # Adiciona as legendas
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    print()
    ax.set_xlim(0, annotation_list[-1][1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Annotations')
    ax.set_title(film_name)
    
    # Remove os ticks do eixo y
    ax.tick_params(axis='y', which='both', length=0)
    plt.title(film_name)

    plt.show()

    
if __name__ == '__main__':
    vsd_gen = VSD_DatasetGenerator(path='C:/Users/CSANT321/Documents/TCC/violence-detection-acoustic-scenes/Datasets/VSD_2014_December_official_release/Hollywood-dev/annotations')
    movie_annotations = vsd_gen.get_movie_original_annotations()
    
    import json
    
    with open("src/dataset_analysis/vsd_annotations.json", "w") as f:
        json.dump(movie_annotations, f)
        
    plot_film_annotations('Armageddon', movie_annotations)