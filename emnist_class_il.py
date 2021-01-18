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
from utils import freeze, new_classifier, new_processor, apply_transformations

print("this script does not take any command-line argument")

dim = 28 ** 2
latent_dim = 3
n_classes = 7

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.15], std=[0.3]),
])
prefix = os.path.join(gettempdir(), "emnist_")
train_set1 = datasets.EMNIST(prefix+"training1", download=True, train=True, transform=tr, split="digits")
train_set2 = datasets.EMNIST(prefix+"training2", download=True, train=True, transform=tr, split="letters")
test_set = datasets.EMNIST(prefix+"testing", download=True, train=False, transform=tr, split="letters")

# train the classifier with a disposable processor
train_on_digits = set([2, 3, 4, 5])  # not symmetric
classifier = new_classifier(latent_dim, n_classes)
digit_processors = [new_processor(dim, latent_dim) for _ in range(10)]
parameters = list(classifier.parameters())
for digit, digit_processor in enumerate(digit_processors):
    if digit in train_on_digits:
        parameters += list(digit_processor.parameters())
optimizer = torch.optim.Adam(parameters)
loss_func = nn.CrossEntropyLoss()
transformations = list(range(n_classes))
progress_bar = tqdm(range(1), total=1, desc="step 1/3")
for epoch in progress_bar:
    loader = torch.utils.data.DataLoader(train_set1, batch_size=1, shuffle=True)
    for images, labels in loader:
        digit = labels[0].item()
        if digit not in train_on_digits:
            continue
        batch = images.expand(n_classes, -1, -1, -1)
        transformed = apply_transformations(batch, transformations)
        latent = digit_processors[digit](transformed.view(n_classes, dim))
        y_pred = classifier(latent)
        loss = loss_func(y_pred, torch.LongTensor(transformations))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    progress_bar.set_description("step 1/3 loss=%.3f" % loss.item())

freeze(classifier)

# prepare the testing data
n_letters = 26
test_images = []
for letter in tqdm(range(n_letters), total=n_letters, desc="step 2/3"):
    loader = torch.utils.data.DataLoader(test_set, batch_size=4096, shuffle=True)
    for images, labels in loader:
        filtering = [i for i, l in enumerate(labels) if l.item() == letter+1]
        images = images[filtering].clone()
        test_images.append(images)
        break

# continual learning loop
history = EvalHistoryFile("hist_emnist_class_il")
processors = []
progress_bar = tqdm(range(n_letters), total=n_letters, desc="step 3/3")
for letter in progress_bar:
    # train a new processor for the current letter
    processor = new_processor(dim, latent_dim)
    processors.append(processor)
    optimizer = torch.optim.Adam(processor.parameters())
    loader = torch.utils.data.DataLoader(train_set2, batch_size=1, shuffle=True)
    for images, labels in loader:
        if labels[0].item() != letter+1:
            continue
        batch = images.expand(n_classes, -1, -1, -1)
        transformed = apply_transformations(batch, transformations)
        latent = processor(transformed.view(n_classes, dim))
        y_pred = classifier(latent)
        loss = loss_func(y_pred, torch.LongTensor(transformations))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    progress_bar.set_description("step 3/3 loss=%.3f" % loss.item())

    # evaluation on all the letters seen so far
    with torch.no_grad():
        hits = 0
        total_images = 0
        for letter in range(len(processors)):
            images = test_images[letter]
            n_images = images.size(0)
            votes = np.zeros((n_images, len(processors)))
            for y_true in range(n_classes):
                transformed = apply_transformations(images, [y_true] * n_images)
                features = transformed.view(n_images, dim)
                # make predictions with each processor
                for candidate, processor in enumerate(processors):
                    pred = torch.log_softmax(classifier(processor(features)), dim=1)[:, y_true]
                    votes[:, candidate] += pred.numpy()

            # choose processor with highest vote
            best = votes.argmax(axis=1)
            hits += (best == letter).sum()
            total_images += n_images

        acc = hits / total_images
        history.log(value=acc, label="task %i" % len(processors))
