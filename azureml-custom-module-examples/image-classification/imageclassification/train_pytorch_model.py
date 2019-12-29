import fire
import os
from azureml.designer.model.io import save_pytorch_state_dict_model
from azureml.designer.model.model_spec.task_type import TaskType
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.logger import logger
from azureml.studio.core.io.model_directory import load_model_from_directory, pickle_loader
from . import modellib

# Disable parallel training to work around built-in score bug.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def entrance(input_model_path='/mnt/chjinche/test_data/init_model',
             train_data_path='/mnt/chjinche/test_data/transform_test/',
             valid_data_path='/mnt/chjinche/test_data/transform_test/',
             save_model_path='/mnt/chjinche/test_data/saved_model',
             epochs=1,
             batch_size=16,
             learning_rate=0.001,
             random_seed=231,
             patience=2):
    logger.info("Start training.")
    logger.info(f"data path: {train_data_path}")
    logger.info(f"data path: {valid_data_path}")
    # TODO: load from schema in case of different tasks
    train_set = ImageDirectory.load(train_data_path).to_torchvision_dataset()
    logger.info(f"Training classes: {train_set.classes}")
    valid_set = ImageDirectory.load(valid_data_path).to_torchvision_dataset()
    # TODO: assert the same classes between train_set and valid_set.
    logger.info("Made dataset")
    classes = train_set.classes
    num_classes = len(classes)
    # TODO: use image directory api to get id-to-class mapping.
    id_to_class_dict = {i: classes[i] for i in range(num_classes)}
    logger.info("Start building model.")
    model_config = load_model_from_directory(input_model_path,
                                             model_loader=pickle_loader).data
    model_class = getattr(modellib, model_config.get('model_class', None),
                          None)
    logger.info(f'Model class: {model_class}.')
    model_config.pop('model_class', None)
    model_config['num_classes'] = num_classes
    logger.info(f'Model_config: {model_config}.')
    model = model_class(**model_config)
    logger.info("Built model. Start training.")
    model.train(train_set=train_set,
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
        "dependencies": [{
            "pip": [
                "azureml-defaults",
                "azureml-designer-core[image]==0.0.25.post7964938",
                "fire==0.1.3",
                "git+https://github.com/StudioCommunity/CustomModules-1.git@master#subdirectory=azureml-custom-module-examples/image-classification",
                "--extra-index-url=https://azureml-modules:3nvdtawseij7o2oenxojj35c43i5lu2ucf77pugohh4g5eqn6xnq@msdata.pkgs.visualstudio.com/_packaging/azureml-modules%40Local/pypi/simple/"
            ]
        }]
    }
    save_pytorch_state_dict_model(model.model,
                                  init_params=model_config,
                                  path=save_model_path,
                                  task_type=TaskType.MultiClassification,
                                  label_map=id_to_class_dict,
                                  conda=conda)
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
