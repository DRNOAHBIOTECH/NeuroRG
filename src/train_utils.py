import time
import copy
import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torchvision.models import densenet201, inception_v3, resnet50, swin_v2_t
from efficientnet_pytorch import EfficientNet

class TrainingModule:
    def __init__(self):
        pass

    def train_model(model, criterion, optimizer, scheduler, plate_idx, dataloaders, dataset_sizes, device, target_addr, model_name, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        PATH = f"{target_addr}/Parameters/state_dict"
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast(enabled=phase == 'train'):
                            outputs = model(inputs)
                            if isinstance(outputs, tuple):  # Inception v3
                                outputs = outputs[0]
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        if isinstance(model, nn.DataParallel):
            torch.save(copy.deepcopy(model.module.state_dict()), f'{PATH}_{model_name}_{plate_idx}')
        else:
            torch.save(copy.deepcopy(model.state_dict()), f'{PATH}_{model_name}{plate_idx}')

        return model