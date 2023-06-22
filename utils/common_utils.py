from collections import OrderedDict
import time
import logging

import torch
import torchio as tio
import pandas as pd

from datasets import Dataset

logger = logging.getLogger(__name__)


class TrainDataset(torch.utils.data.Dataset):
    """
    build training dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels):
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        return self.dict[idx]


class ValidationDataset(torch.utils.data.Dataset):
    """
    build validation dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels):
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        return self.dict[idx]


class TestDataset(torch.utils.data.Dataset):
    """
    build test dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels):
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        return self.dict[idx]


def transform_to_huggingface_dataset(pt_dataset):
    huggingface_dataset = Dataset.from_generator(gen, gen_kwargs={"pt_dataset": pt_dataset})
    return huggingface_dataset


def gen(pt_dataset):
    for idx in range(len(pt_dataset)):
        yield pt_dataset[idx]


class RunManager:
    """capture model stats"""
    def __init__(self):
        self.epoch_num_count = 0
        self.epoch_start_time = None

        # train/validation/test metrics
        self.train_epoch_loss = 0
        self.train_epoch_standard_loss = 0
        self.validation_epoch_loss = 0
        self.test_epoch_loss = 0
        self.run_correlation_train = []
        self.run_correlation_validation = []
        self.run_correlation_test = []

        # run data saves the stats from train/validation/test set
        self.run_data = []
        self.run_start_time = None

        self.train_data_loader = None
        self.validation_data_loader = None
        self.test_data_loader = None
        self.epoch_stats = None

    def begin_run(self, train_data_loader, validation_data_loader, test_data_loader):
        self.run_start_time = time.time()
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.test_data_loader = test_data_loader
        logger.info('Begin Run!')

    def end_run(self):
        self.epoch_num_count = 0
        logger.info('End Run!')

    def begin_epoch(self):
        self.epoch_num_count += 1
        self.epoch_start_time = time.time()
        # initialize metrics
        self.train_epoch_loss = 0
        self.train_epoch_standard_loss = 0
        self.validation_epoch_loss = 0
        self.test_epoch_loss = 0
        logger.info(f'Start epoch {self.epoch_num_count}')

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        # calculate metrics
        train_loss = self.train_epoch_loss / len(self.train_data_loader.dataset)
        train_standard_loss = self.train_epoch_standard_loss / len(self.train_data_loader.dataset)
        validation_loss = self.validation_epoch_loss / len(self.validation_data_loader.dataset)
        test_loss = self.test_epoch_loss / len(self.test_data_loader.dataset)
        logger.info(f'End epoch {self.epoch_num_count}')

        # add stats from current epoch to run data
        self.epoch_stats = OrderedDict()
        self.epoch_stats['epoch'] = self.epoch_num_count
        self.epoch_stats['train_loss'] = float(f'{train_loss:.2f}')
        self.epoch_stats['train_standard_loss'] = float(f'{train_standard_loss:.2f}')
        self.epoch_stats['validation_loss'] = float(f'{validation_loss:.2f}')
        self.epoch_stats['test_loss'] = float(f'{test_loss:.2f}')
        self.epoch_stats['train_correlation'] = float(f'{self.run_correlation_train[-1]:.2f}')
        self.epoch_stats['validation_correlation'] = float(f'{self.run_correlation_validation[-1]:.2f}')
        self.epoch_stats['test_correlation'] = float(f'{self.run_correlation_test[-1]:.2f}')
        self.epoch_stats['epoch_duration'] = float(f'{epoch_duration:.1f}')
        self.epoch_stats['run_duration'] = float(f'{run_duration:.1f}')
        self.run_data.append(self.epoch_stats)

    def track_train_loss(self, loss):
        # accumulate training loss for all batches
        self.train_epoch_loss += loss.item() * self.train_data_loader.batch_size

    def track_standard_train_loss(self, loss):
        # accumulate training loss for all batches(standard loss)
        self.train_epoch_standard_loss += loss.item() * self.train_data_loader.batch_size

    def track_validation_loss(self, loss):
        # accumulate validation loss for all batches
        self.validation_epoch_loss += loss.item() * self.validation_data_loader.batch_size

    def track_test_loss(self, loss):
        # accumulate test loss for all batches
        self.test_epoch_loss += loss.item() * self.test_data_loader.batch_size

    def collect_train_correlation(self, correlation):
        self.run_correlation_train.append(correlation)

    def collect_validation_correlation(self, correlation):
        self.run_correlation_validation.append(correlation)

    def collect_test_correlation(self, correlation):
        self.run_correlation_test.append(correlation)

    def display_epoch_results(self):
        # display stats from the current epoch
        logger.info(self.epoch_stats)

    def save(self, filename):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')
