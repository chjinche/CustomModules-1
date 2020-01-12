import fire
import torch.nn as nn
from azureml.studio.core.logger import logger
from azureml.studio.core.io.model_directory import save_model_to_directory, pickle_dumper
from ..common.basenet import BaseNet


class DenseNet(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_nn()

    # def update_nn(self):
    #     if self.pretrained:
    #         num_classes = self.kwargs.get('num_classes', None)
    #         num_final_in = self.model.classifier.in_features
    #         self.model.classifier = nn.Linear(num_final_in, num_classes)
    # def update_nn(self):
    #     if self.pretrained:
    #         num_classes = self.kwargs.get('num_classes', None)
    #         num_final_out = self.model.classifier.out_features
    #         self.model.add_module('fc', nn.Linear(num_final_out, num_classes))
    def update_nn(self):
        if self.pretrained:
            num_classes = self.kwargs.get('num_classes', None)
            num_final_out = self.model.classifier.out_features
            self.model2 = nn.Linear(num_final_out, num_classes)

    def forward(self, x):
        return self.model2(self.model(x)) if self.pretrained else self.model(x)


def entrance(save_model_path='/mnt/chjinche/test_data/init_model',
             model_type='densenet201',
             pretrained=True,
             memory_efficient=False):
    model_config = {
        'model_class': 'DenseNet',
        'model_type': model_type,
        'pretrained': pretrained,
        'memory_efficient': memory_efficient,
        'trainer': 'ClassificationTrainer',
    }
    logger.info('Dump untrained model.')
    logger.info(f'Model config: {model_config}.')
    dumper = pickle_dumper(model_config, 'model_config.pkl')
    save_model_to_directory(save_model_path, dumper)
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(entrance)
