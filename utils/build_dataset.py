import logging, pickle

from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset
import numpy as np
import pandas as pd

from utils.common_utils import TrainDataset, ValidationDataset, TestDataset

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results'


def build_dataset(args):
    """main function for dataset building"""
    if args.dataset == 'camcan':
        dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age = build_dataset_camcan(args)

    # update arguments
    args.age_limits = lim
    args.input_shape = input_shape
    args.median_age = median_age
    return dataset_train, dataset_validation, dataset_test, args


# ------------------- Cam-CAN Dataset ---------------------
def build_dataset_camcan(args):
    """load Cam-Can MRI data: https://www.cam-can.org """
    # load MRI data
    images, df = pickle.load(open(args.data_dir, 'rb'))
    # reformat DataFrame
    df = df.reset_index()
    # retrieve the minimum, maximum and median age for the skewed loss
    lim = (df['Age'].min(), df['Age'].max())
    median_age = df['Age'].median()
    # add color channel for images (bs, H, D, W) -> (bs, 1, H, D, W)
    images = np.expand_dims(images, axis=1)

    assert len(images.shape) == 5, images.shape
    assert images.shape[1] == 1, images.shape[1]
    assert len(images) == len(df), len(images)

    # assign a categorical label to Age for Stratified Split
    df['Age_categorical'] = pd.qcut(df['Age'], 25, labels=[i for i in range(25)])

    # Stratified train validation-test Split
    split = StratifiedShuffleSplit(test_size=args.val_test_size, random_state=args.random_state)
    train_index, validation_test_index = next(split.split(df, df['Age_categorical']))
    stratified_validation_test_set = df.loc[validation_test_index]
    assert sorted(train_index.tolist() + validation_test_index.tolist()) == list(range(len(df)))

    # Stratified validation test Split
    split2 = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    validation_index, test_index = next(split2.split(stratified_validation_test_set,
                                                     stratified_validation_test_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(validation_index.tolist() + test_index.tolist()) == \
        list(range(len(stratified_validation_test_set.index)))
    assert sorted(validation_index.tolist() + test_index.tolist()) != \
        sorted(list(stratified_validation_test_set.index))

    # get the correct index of original DataFrame for validation/test set
    validation_index = validation_test_index[validation_index]
    test_index = validation_test_index[test_index]

    # ensure there is no duplicated index
    assert sorted(train_index.tolist() + validation_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # get train/validation/test set
    train_images = images[train_index].astype(np.float32)
    validation_images = images[validation_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    # add dimension for labels: (32,) -> (32, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    validation_labels = np.expand_dims(df.loc[validation_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    logger.info(f'Training images shape: {train_images.shape}, validation images shape: {validation_images.shape}, '
                f'testing images shape: {test_images.shape}, training labels shape: {train_labels.shape}, '
                f'validation labels shape: {validation_labels.shape}, testing labels shape: {test_labels.shape}')

    # Huggingface Dataset for train set.
    dataset_train = TrainDataset(images=train_images, labels=train_labels)
    dataset_train = transform_to_huggingface_dataset(pt_dataset=dataset_train)
    dataset_train.set_format(type='torch', columns=['image', 'label'])
    # Huggingface Dataset for validation set
    dataset_validation = ValidationDataset(images=validation_images, labels=validation_labels)
    dataset_validation = transform_to_huggingface_dataset(pt_dataset=dataset_validation)
    dataset_validation.set_format(type='torch', columns=['image', 'label'])
    # Huggingface Dataset for test set
    dataset_test = TestDataset(images=test_images, labels=test_labels)
    dataset_test = transform_to_huggingface_dataset(pt_dataset=dataset_test)
    dataset_test.set_format(type='torch', columns=['image', 'label'])

    return dataset_train, dataset_validation, dataset_test, lim, train_images.shape[1:], median_age


def transform_to_huggingface_dataset(pt_dataset):
    huggingface_dataset = Dataset.from_generator(gen, gen_kwargs={"pt_dataset": pt_dataset})
    return huggingface_dataset


def gen(pt_dataset):
    for idx in range(len(pt_dataset)):
        yield pt_dataset[idx]
