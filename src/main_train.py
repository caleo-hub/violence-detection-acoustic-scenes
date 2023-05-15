import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils.dataloader import H5Dataset

# Caminho para os arquivos h5
h5_files = ['src/h5_files/GunshotForensic_feature.h5',
            'src/h5_files/HEAR_Test_feature.h5',
            'src/h5_files/MIVIA_test_feature.h5',
            'src/h5_files/MIVIA_train_feature.h5',
            'src/h5_files/SESA_feature.h5',
            'src/h5_files/vsd_clipped_features.h5'
]

# Carregando os dataloaders
dataset = H5Dataset(h5_files, exclude_classes=[6])
dataloaders = dataset.get_k_fold_data_loaders(batch_size=32,
                                               num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ajustando a ResNet101
model = models.resnet101(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  # ajustar a última camada para 6 classes
model.conv1 = nn.Conv2d(1,
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False)  # ajustar a primeira camada para 1 canal
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Configurando o TensorBoard
writer = SummaryWriter()

# Loop de treinamento
for fold, (train_loader, val_loader) in enumerate(dataloaders):
    print(f'Starting fold {fold + 1}/{len(dataloaders)}')
    for epoch in range(10):  # número de épocas
        print(f'Starting epoch {epoch + 1}/10')

        # Treinamento
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = inputs.unsqueeze(1).to(device)  # Adiciona um canal extra
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validação
        model.eval()
        running_corrects = 0
        f1_scores = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.unsqueeze(1).to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                f1_scores.append(f1_score(labels.data.cpu(), preds.cpu(), average=None))

        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        epoch_f1 = torch.mean(torch.tensor(f1_scores), dim=0)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1}')

        writer.add_scalar('training loss', epoch_loss, epoch)
        writer.add_scalar('accuracy', epoch_acc, epoch)
        writer.add_histogram('F1 score', epoch_f1, epoch)

    print(f'Finished fold {fold + 1}/{len(dataloaders)}')

# Salvando o modelo treinado
torch.save(model.state_dict(), 'src/models/model_resnet101.pth')

# Fechando o TensorBoard writer
writer.close()

print('Finished Training')

