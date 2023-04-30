import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotator:
    def __init__(self, window_size, hop_length):
        self.window_size = window_size
        self.hop_length = hop_length
    
    def partition_film(self, annotations, film_name):
        self.film_name = film_name
        annotation_list = annotations[self.film_name]
        new_annotations = []
        start_time = 0
        end_time = self.window_size
        current_window_annotations = []
        
        while end_time <= annotation_list[-1][1]:
            for ann in annotation_list:
                if start_time <= ann[1] < end_time:
                    if ann[2] != '(nothing)' and ann[2] not in current_window_annotations:
                        current_window_annotations.append(ann[2])
                    elif ann[2] == '(nothing)' and len(current_window_annotations) == 0:
                        current_window_annotations.append(ann[2])

            
            # Remove '(nothing)' se estiver acompanhada de outras anotações diferentes
            if '(nothing)' in current_window_annotations and len(set(current_window_annotations)) > 1:
                current_window_annotations.remove('(nothing)')
            if len(current_window_annotations) == 0:
                current_window_annotations.append('(nothing)')
            
            new_annotations.append((start_time, end_time, current_window_annotations))
            
            start_time += self.hop_length
            end_time += self.hop_length
            current_window_annotations = []
        
        return new_annotations
    
    def plot_timeline(self, annotation_list):
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.set_ylim([0, 1])
        ax.set_xlim([0, annotation_list[-1][1]])

        # Define as cores para cada combinação de anotações 
        colors = {}
        # Plota as anotações
        for start_time, end_time, annotation in annotation_list:
            
            if tuple(sorted(annotation)) not in colors.keys():
                colors[tuple(sorted(annotation))] = plt.cm.tab20(len(colors.keys()) + 1)
                
            color = colors[tuple(sorted(annotation))]
            rect = Rectangle((start_time, 0), end_time - start_time, 1, color=color, alpha=1)
            ax.add_patch(rect)
            
        # Cria a legenda
        handles = []
        labels = []
        for key, value in colors.items():
            rect = Rectangle((0, 0), 1, 1, color=value)
            handles.append(rect)
            labels.append(', '.join(key))

        ax.legend(handles, labels, loc='upper right', fontsize='x-small')
        
        plt.title(self.film_name)
        plt.show()
