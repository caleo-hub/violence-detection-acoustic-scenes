# Modelo de aprendizado de máquina para identificação de cenários de violência

Este é o repositório do projeto de construção de um modelo de aprendizado de máquina para a classificação de cenas acústicas envolvendo violência.

## Descrição do projeto:
O objetivo deste projeto é desenvolver um modelo capaz de identificar automaticamente cenas de violência em arquivos de áudio. Para isso, são utilizadas técnicas de processamento de áudio e aprendizado de máquina, com diversos datasets de áudio para treinamento e avaliação.

## Conteúdo do repositório:
Este repositório inclui códigos para todas as etapas do projeto, organizados em pastas de acordo com as etapas: pré-processamento de dados, treinamento de modelos e avaliação de resultados. Além disso, inclui arquivos de documentação e relatórios sobre o projeto.

## Instruções de uso:
Para utilizar os códigos deste repositório, siga os passos:

1. Clone o repositório:

```bash
git clone https://github.com/caleo-hub/violence-detection-acoustic-scenes.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Baixe o modelo treinado
Baixe o modelo treinado no [link](https://drive.google.com/file/d/11OuE2eTB8Y0whF772LtGy_FEKKGK-tGL/view?usp=share_link
)

E coloque o arquivo .pth no folder src/models

## Teste Você Mesmo
Execute o main_test.ipynb para testar com seu próprio áudio


# Datasets

## Classes escolhidas
   * Agressão física
   * Disparo de arma de fogo
   * Grito



- [] Google Audioset
  - Screaming
  - Gunshots, gunfire
  - Slap, smack
  
- [] MIVIA audio events data set for surveillance applications
  - Scream
  - Gunshot
  - Casual
- 
- [] Sound Events for Surveillance Applications (SESA) Dataset
  - Casual (not a threat)
  - Gunshot
  - Explosion

- [] HEAR Dataset
  - Violência Física (socos, tapas)
  - Áudio de Fundo (pets, television, toilet flush, door radio, water, vacuum cleaner, sobbing, noise, sink and fying)
  - Áudio de Primeiro plano (inside, conversation and speech)

- [] Gunshot Audio Forensics Dataset
  - High Standard Sport King [.22LR, Pistol]
  - S&W 34-1 [.22LR, Revolver]
  - Ruger 10/22 [.22LR, Carbine]
  - Remington 33 Bolt-Action Rifle [.22LR, Rifle]
  - Lorcin L380 [.380 Auto, Pistol]
  - S&W 10-8 [.38SPL, Revolver]
  - Ruger Blackhawk [.357 MAG, Revolver]
  - Glock 19 [9mm Luger, Pistol] (Qty 2)
  - Sig P225 [9mm Luger, Pistol]
  - M&P 40 [.40 S&W, Pistol] (Qty 2)
  - HK USP Compact [.40 S&W, Pistol]
  - Glock 21 [.45 Auto, Pistol]
  - Colt 1911 [.45 Auto, Pistol]
  - Kimber Tactical Custom [.45 Auto, Pistol]
  - M16A1 AR15 [.223R/5.56, Rifle]
  - WASR 10/63 AK47 [7.62x39mm, Carbine]
  - Winchester M14 [.308W/7.62, Rifle]
  - Remington 700 [.308W/7.62, Rifle]
  - Rock River LAR-15 [.300 Blackout, Rifle]
  - Russian SKS [7.62x39mm, Pistol]
  - PWS MK107 Mod 1 [7.62×39, Pistol]


