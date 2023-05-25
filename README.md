# Modelo de aprendizado de máquina para identificação de cenários de violência

Este é o repositório do projeto de construção de um modelo de aprendizado de máquina para a classificação de cenas acústicas envolvendo violência.

## Descrição do projeto:
O objetivo deste projeto é desenvolver um modelo capaz de identificar automaticamente cenas de violência em arquivos de áudio. Para isso, são utilizadas técnicas de processamento de áudio e aprendizado de máquina, com diversos datasets de áudio para treinamento e avaliação.

## Conteúdo do repositório:
Este repositório inclui códigos para todas as etapas do projeto, organizados em pastas de acordo com as etapas: pré-processamento de dados, treinamento de modelos e avaliação de resultados. Além disso, inclui arquivos de documentação e relatórios sobre o projeto.

## Instruções de uso:
Para utilizar os códigos deste repositório, siga os passos:

1. Clone o repositório:

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Baixe o modelo treinado
Baixe os modelos treinado no [link](https://drive.google.com/drive/folders/1tdLO9I310XDlEV_5JDY-9zFNaZaYZpFU?usp=share_link
)

E extraia o arquivo .zip na pasta src/models

## Teste Você Mesmo
Execute o main_test.ipynb para testar com seu próprio áudio


# Datasets

## Classes escolhidas
   * Nada (Sem violência)
   * Grito
   * Agressão física
   * Disparo de arma de fogo
   

### Datasets

### Datasets

- Gunshot Audio Forensics Dataset:
  - Ryan Lilien, Jason Housman, Ram Mettu, Todd Weller, Lucien Haag, Michael Haag. "Gunshot Audio Forensics Dataset." 2017. [Link](http://cadreforensics.com/audio/). DOI: [10.1371/journal.pone.0183754](https://doi.org/10.1371/journal.pone.0183754)

- Sound Events for Surveillance Applications Dataset:
  - Tito Spadini. "Sound Events for Surveillance Applications." October 2019. [Link](https://doi.org/10.5281/zenodo.3519845). DOI: [10.5281/zenodo.3519845](https://doi.org/10.5281/zenodo.3519845)

- Google Audioset:
  - Jan Gemmeke, Georg Heigold, Alexander Ewert, Oliver Puhr, Björn Schuller. "Audioset: An ontology and dataset for audio events." Proceedings of the 2017 ACM on Multimedia Conference, 2017. [Link](https://research.google.com/audioset/)

- MIVIA Audio Events Dataset:
  - Pasquale Foggia, Mario Vento, Gennaro Percannella, Pierluigi Ritrovato, Alessia Saggese, Luca Greco, Vincenzo Carletti, Antonio Greco. "Mivia Audio Events Dataset." 2016. [Link](https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/)

- HEAR Dataset:
  - Tiago Lacerda, Péricles Miranda, André Câmara, Ana Furtado. "Deep Learning and Mel-spectrograms for Physica Violence Detection in Audio." Anais do XVIII Encontro Nacional de Inteligência Artificial e Computacional, 2021. [Link](https://sol.sbc.org.br/index.php/eniac/article/view/18259). DOI: [10.5753/eniac.2021.18259](https://doi.org/10.5753/eniac.2021.18259)

- Violent Scenes Dataset:
  - C.H. Demarty, C. Penet, M. Soleymani, G. Gravier. "VSD, a public dataset for the detection of violent scenes in movies: design, annotation, analysis and evaluation." Multimedia Tools and Applications, 2014. [Link](https://www.interdigital.com/data_sets/violent-scenes-dataset)

- AudioSet Processing:
  - Aoife McDonagh. "AudioSet Processing." 2023. [Link](https://github.com/aoifemcdonagh/audioset-processing)


