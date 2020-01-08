import fire
import os
from azureml.designer.model.io import save_pytorch_state_dict_model
from azureml.designer.model.model_spec.task_type import TaskType
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.image_schema import ImageAnnotationTypeName
from azureml.studio.core.logger import logger
from azureml.studio.core.io.model_directory import load_model_from_directory, pickle_loader
from .. import initialize_models
from .utils import ConvertCocoPolysToMask

# TODO:Find idle device rather than hard code. Disable parallel training to work around built-in score bug.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def entrance(input_model_path='/mnt/chjinche/test_data/detection/init_model',
             train_data_path='/mnt/chjinche/test_data/detection/transform/',
             valid_data_path='/mnt/chjinche/test_data/detection/transform',
             save_model_path='/mnt/chjinche/test_data/detection/saved_model',
             epochs=1,
             batch_size=32,
             learning_rate=0.001,
             random_seed=231,
             patience=2):
    logger.info("Start training.")
    logger.info(f"data path: {train_data_path}")
    logger.info(f"data path: {valid_data_path}")
    # TODO: load from schema in case of different tasks
    ann_type = ImageDirectory.load(train_data_path).get_annotation_type()
    logger.info(f'task type: {ann_type}')
    transforms = ConvertCocoPolysToMask(
    ) if ann_type == ImageAnnotationTypeName.OBJECT_DETECTION else None
    train_set = ImageDirectory.load(train_data_path).to_torchvision_dataset(
        transforms=transforms)
    # logger.info(f"Training classes: {train_set.classes}")
    valid_set = ImageDirectory.load(valid_data_path).to_torchvision_dataset(
        transforms=transforms)
    # TODO: assert the same classes between train_set and valid_set.
    logger.info("Made dataset")
    # classes = train_set.categories
    num_classes = train_set.num_of_classes
    class_to_idx = train_set.class_to_idx
    # print(classes)
    # num_classes = len(classes)
    # num_classes = 91
    # TODO: use image directory api to get id-to-class mapping.
    id_to_class_dict = dict((v, k) for k, v in class_to_idx.items())
    logger.info(f'{num_classes} classes, {class_to_idx}')
    # id_to_class_dict = {i: classes[i] for i in range(num_classes)}
    logger.info("Start building model.")
    model_config = load_model_from_directory(input_model_path,
                                             model_loader=pickle_loader).data
    model_class = getattr(initialize_models,
                          model_config.get('model_class', None), None)
    logger.info(f'Model class: {model_class}.')
    model_config.pop('model_class', None)
    model_config['num_classes'] = num_classes
    logger.info(f'Model_config: {model_config}.')
    model = model_class(**model_config)
    logger.info("Built model. Start training.")
    model.fit(train_set=train_set,
              valid_set=valid_set,
              epochs=epochs,
              batch_size=batch_size,
              lr=learning_rate,
              random_seed=random_seed,
              patience=patience)
    # Save model file, configs and install dependencies
    # TODO: designer.model could support pathlib.Path
    # local_dependencies = [str(Path(__file__).parent.parent)]
    # logger.info(f'Ouput local dependencies {local_dependencies}.')

    conda = {
        "name": "project_environment",
        "channels": ["defaults"],
        "dependencies": [
            "python=3.6.8",
            "cython=0.29.14",
            "numpy=1.16.4", {
                "pip": [
                    "azureml-defaults",
                    "azureml-designer-core[image]==0.0.26.post8829093",
                    "torch==1.3",
                    "torchvision==0.4.1",
                    "fire==0.1.3",
                    "pycocotools==2.0.0",
                    "git+https://github.com/microsoft/ComputerVision.git@master#egg=utils_cv",
                    "git+https://github.com/chjinche/CustomModules-1.git@master#subdirectory=azureml-custom-module-examples/pytorch-modules",
                    "--extra-index-url=https://azureml-modules:3nvdtawseij7o2oenxojj35c43i5lu2ucf77pugohh4g5eqn6xnq@msdata.pkgs.visualstudio.com/_packaging/azureml-modules%40Local/pypi/simple/"
                ]
            }
        ]
    }
    save_pytorch_state_dict_model(model,
                                  init_params=model_config,
                                  path=save_model_path,
                                  task_type=TaskType.MultiClassification,
                                  label_map=id_to_class_dict,
                                  conda=conda)
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
