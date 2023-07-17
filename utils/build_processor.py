import logging, torch, time
import torchio as tio

import config

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results'


def build_processor(dataset_train, dataset_val, dataset_test):
    """main function for dataset preprocessing"""
    config.train_transforms, config.val_transforms = medical_augmentation_hf()
    dataset_train.set_transform(preprocess_train)
    dataset_val.set_transform(preprocess_val)
    dataset_test.set_transform(preprocess_val)

    # set a checker to check transformed data
    idx = len(dataset_train) // 2
    logger.info(f"The type of model input is {type(dataset_train[idx]['image'])}, "
                f"output is {type(dataset_train[idx]['label'])}. "
                f"The shape of model input is {dataset_train[idx]['image'].shape}")

    return dataset_train, dataset_val, dataset_test


def medical_augmentation_pt(images):
    training_transform = tio.Compose([
        tio.RandomBlur(p=0.5),  # blur 50% of times
        tio.RandomNoise(p=0.5),  # Gaussian noise 50% of times
        tio.RandomFlip(flip_probability=0.5),
    ])
    return training_transform(images)



def medical_augmentation_hf():
    """define data augmentation pipeline"""
    train_transform = tio.Compose([
        tio.RandomBlur(p=0.5),  # blur 50% of times
        tio.RandomNoise(p=0.5),  # Gaussian noise 50% of times
        tio.RandomFlip(flip_probability=0.5),
        tio.ZNormalization()
    ])

    val_transform = tio.Compose([
        tio.ZNormalization()
    ])

    return train_transform, val_transform


def preprocess_train(example_batch):
    """
    Apply train_transforms across a batch.
    Note that set_transform will replace the format defined by set_format and
        will be applied on-the-fly (when it is called).
    """
    example_batch["image"] = [config.train_transforms(torch.tensor(image)) for image in example_batch["image"]]
    example_batch["label"] = [torch.tensor(label) for label in example_batch["label"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["image"] = [config.val_transforms(torch.tensor(image)) for image in example_batch["image"]]
    example_batch["label"] = [torch.tensor(label) for label in example_batch["label"]]
    return example_batch
