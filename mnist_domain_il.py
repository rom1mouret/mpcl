#!/usr/bin/env python3

import os
from tempfile import gettempdir
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch


from eval_history import EvalHistoryFile
from utils import freeze, confidence, new_classifier, new_processor

print("this script does not take any command-line argument")

dim = 28 ** 2
latent_dim = 3

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.15], std=[0.3]),
])
prefix = os.path.join(gettempdir(), "mnist_")
train_set = datasets.MNIST(prefix+"training", download=True, train=True, transform=tr)
test_set = datasets.MNIST(prefix+"testing", download=True, train=False, transform=tr)

# train the classifier with a disposable processor
classifier = new_classifier(latent_dim, 10)
processor = new_processor(dim, latent_dim)
parameters = list(processor.parameters()) + list(classifier.parameters())
optimizer = torch.optim.Adam(parameters)
loss_func = nn.CrossEntropyLoss()
for epoch in tqdm(range(3), total=3):
    loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    for images, labels in loader:
        features = images.view(images.size(0), dim)
        latent = processor(features)
        y_pred = classifier(latent)
        loss = loss_func(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

freeze(classifier)

# prepare the testing data
loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)
for images, test_labels in loader:
    test_features = images.view(images.size(0), dim)
    break

# create 40 tasks
tasks = [np.random.permutation(dim) for _ in range(40)]

# continual learning loop
history = EvalHistoryFile("hist_mnist_domain_il")
processors = []
progress_bar = tqdm(enumerate(tasks), total=len(tasks))
for task_i, task in progress_bar:
    # train a new processor for the current task
    processor = new_processor(dim, latent_dim)
    processors.append(processor)
    optimizer = torch.optim.Adam(processor.parameters())
    loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    for images, labels in loader:
        features = images.view(images.size(0), dim)[:, task]
        latent = processor(features)
        y_pred = classifier(latent)
        loss = loss_func(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_description("loss: %.4f" % loss.item())

    # evaluation on all the tasks seen so far
    with torch.no_grad():
        hits = 0
        for task_j in range(len(processors)):
            permuted = test_features[:, tasks[task_j]]
            # make predictions with each processor
            pred = [classifier(processor(permuted)) for processor in processors]
            # pick out the processor with the most confidence
            scores = torch.cat([confidence(p) for p in pred], dim=1)
            _, best = scores.max(dim=1)
            for i, k in enumerate(best):
                _, y_pred = pred[k][i].max(dim=0)
                if y_pred == test_labels[i]:
                    hits += 1

        acc = hits / (len(processors) * test_features.size(0))
        history.log(value=acc, label="task %i" % len(processors))
