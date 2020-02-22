# https://raw.githubusercontent.com/galatolofederico/pytorch-balanced-batch/master/sampler.py
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random
import torch


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None,shuffle=False):
        self.labels = labels
        self.dataset = dict() # keys are class labels, values are set of sample indices associated with each label
        self.balanced_max = 0
        self.shuffle = shuffle
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0 # keeps track of which class should be sampled
        self.indices = [-1] * len(self.keys) # keeps track of number of samples per class
        print('balanced_max: ',self.balanced_max)
        print('number of samples in balanced dataset {}'.format(self.balanced_max*len(self.keys)))
        # print(self.indices)

    def __iter__(self):
        if self.shuffle:
            print('shuffling dataset')
            for label in self.dataset:
                random.shuffle(self.dataset[label])
        # print(self.dataset)

        while self.indices[self.currentkey] < self.balanced_max - 1:
            # print(self.indices)
            # print('curr_key',self.currentkey)
            self.indices[self.currentkey] += 1
            # print('indices',self.indices)
            # print('yielding',self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]])
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys) # I geuss an assertion that currentkey stays between 0 and num_class-1?
        self.indices = [-1] * len(self.keys)

        # print(self.indices)
        # print(self.currentkey)

    def _get_label(self, dataset, idx):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


if __name__=='__main__':
    epochs = 1
    size = 10
    features = 5
    classes_prob = torch.tensor([0.1, 0.4, 0.5])

    dataset_X = torch.randn(size, features)
    dataset_Y = torch.distributions.categorical.Categorical(classes_prob.repeat(size, 1)).sample()
    print('dataset', dataset_X)
    print('label', dataset_Y)
    dataset = torch.utils.data.TensorDataset(dataset_X, dataset_Y)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=BalancedBatchSampler(dataset, dataset_Y), batch_size=6)

    for epoch in range(0, epochs):
        for batch_x, batch_y in train_loader:
            # pass
            print("epoch: %d labels: %s\ninputs: %s\n" % (epoch, batch_y, batch_x))