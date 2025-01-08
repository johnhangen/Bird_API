from configs.config import Config

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, criterion, optimizer, config: Config):
    for epoch in range(config.Train.Epoch):
        model.train()

        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)

            optimizer.zero_grad()

            outputs= model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    return model

