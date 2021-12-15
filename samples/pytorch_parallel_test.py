# import copy
import torch
from torchvision import datasets, models, transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import hpopt

from multiprocessing import Process

def my_trainer(config, device='cpu'):
    lr = config['params']['lr']
    bs = config['params']['bs']

    print(f'train model start! lr : {lr} / bs : {bs}')

    transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize((0.5,), (0.5,), )
            ])

    trainset = datasets.SVHN('./dataset/files',
                             split ='train', 
                             transform=transform, 
                             download=False)
    
    trainset = hpopt.createHpoDataset(trainset, config)

    trainset, validset = torch.utils.data.random_split(trainset, 
            [len(trainset) - int(len(trainset)*0.2), int(len(trainset)*0.2)])

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr)

    critic = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)

    data_loader = {'train' : DataLoader(trainset, batch_size=bs, shuffle = True),
                   'val'   : DataLoader(validset, batch_size=bs, shuffle = True)}

    for current_epoch in range(config["iterations"]):
        print(f'\rEpoch {current_epoch+1} / {config["iterations"]}')

        epoch_loss = 0.0
        epoch_acc = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            idx = 0
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = critic(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        print(f'{idx/len(data_loader[phase])*100:.2f}%', end='')
                        idx += 1
                        print('\r', end='')

                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (len(trainset) if phase == 'train' else len(validset))
            epoch_acc = running_correct.double() / (len(trainset) if phase == 'train' else len(validset))
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if hpopt.report(config=config, score=epoch_acc.item()) == hpopt.Status.STOP:
            break

full_dataset = datasets.SVHN('./dataset/files', 
                         split ='train', 
                         download=True)

hp_configs = {"lr": hpopt.search_space("loguniform", [0.0001, 0.1]),
              "bs": hpopt.search_space("qloguniform", [8, 128, 4])}

my_hpo = hpopt.create(save_path='./tmp/my_hpo_resnet',
                      search_alg="bayes_opt",
                      search_space=hp_configs,
                      ealry_stop="median_stop",
                      num_init_trials=5,
                      #num_trials=20,
                      #max_iterations=2,
                      #subset_ratio=1.0,
                      expected_time_ratio=1,
                      num_full_iterations=10,
                      full_dataset_size=len(full_dataset))

while True:
    configs = my_hpo.get_next_samples()

    if len(configs) == 0:
        break

    proc_list = []
    for config in configs:
        p = Process(target=my_trainer, args=(config, 'cuda',))
        proc_list.append(p)
        p.start()

    for p in proc_list:
        p.join()

print("best hp: ", my_hpo.get_best_config())
