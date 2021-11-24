import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import copy



def _maybe_download(path):
    return datasets.MNIST(root = path, 
                   train = True, 
                   download = True)


def load_torch_loader(path_to_download,BATCH_SIZE = 64):


    # download and extract the data
    train_data = _maybe_download(path_to_download)

    mean = train_data.data.float().mean() / 255
    std = train_data.data.float().std() / 255

    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    # split training dataset into validation set
    train_data, valid_data = data.random_split(train_data, 
                                           [n_train_examples, n_valid_examples])


    # create data augmentation in dataset
    train_transforms = transforms.Compose([
                            transforms.RandomRotation(5, fill=(0,)),
                            transforms.RandomCrop(28, padding = 2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [mean], std = [std])
                      ])

    test_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean = [mean], std = [std])
                     ])

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    train_data = datasets.MNIST(root = path_to_download, 
                                train = True, 
                                download = True, 
                                transform = train_transforms)

    test_data = datasets.MNIST(root = path_to_download, 
                               train = False, 
                               download = True, 
                               transform = test_transforms)

    train_iterator = data.DataLoader(train_data, 
                                     shuffle = True, 
                                     batch_size = BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data, 
                                     batch_size = BATCH_SIZE)

    test_iterator = data.DataLoader(test_data, 
                                    batch_size = BATCH_SIZE)
    return train_iterator, valid_iterator, test_iterator


