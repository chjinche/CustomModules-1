import fire
import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from azureml.studio.core.logger import logger
from utils_cv.detection.references.engine import train_one_epoch, evaluate
from utils_cv.common.gpu import torch_device
from utils_cv.detection.model import _calculate_ap
from utils_cv.detection.references import utils
from azureml.studio.core.io.model_directory import save_model_to_directory, pickle_dumper
from ..common.basenet import BaseNet
# from .utils import remove_images_without_annotations


class FasterRCNN(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_nn()
        logger.info(f"Model init finished, {self.model}.")

    def update_nn(self):
        if self.pretrained:
            num_classes = self.kwargs.get('num_classes', None)
            # num_classes = 91
            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            # that has num_classes which is based on the dataset
            # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
            # TODO:hard code num_classes for COCO now
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def fit(self,
            train_set,
            valid_set,
            epochs,
            batch_size,
            lr=0.001,
            wd=0.0001,
            momentum=0.9,
            random_seed=None,
            patience=10):
        logger.info('Torch cuda random seed setting.')
        # Torch cuda random seed setting
        if random_seed is not None:
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    torch.cuda.manual_seed_all(random_seed)
                else:
                    torch.cuda.manual_seed(random_seed)
            else:
                torch.manual_seed(random_seed)
        logger.info("Data start loading.")
        # DataLoader
        # train_set = remove_images_without_annotations(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=utils.collate_fn,
            pin_memory=(torch.cuda.is_available()),
            num_workers=0)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
            pin_memory=(torch.cuda.is_available()),
            num_workers=0)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if torch.cuda.device_count() > 1:
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model).cuda()
            self.model = torch.nn.DataParallel(self.model).cuda()

        # reduce learning rate every step_size epochs by a factor of gamma (by default) 0.1.
        step_size = None
        if step_size is None:
            step_size = int(np.round(epochs / 1.5))

        self.device = torch_device()
        # construct our optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=wd
        )

        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=0.1
        )

        # store data in these arrays to plot later
        self.losses = []
        self.ap = []
        self.ap_iou_point_5 = []

        # main training loop
        self.epochs = epochs
        for epoch in range(self.epochs):

            # train for one epoch, printing every 10 iterations
            logger_per_epoch = train_one_epoch(
                self.model,
                self.optimizer,
                train_loader,
                self.device,
                epoch,
                print_freq=10,
            )
            self.losses.append(logger_per_epoch.meters["loss"].median)

            # update the learning rate
            self.lr_scheduler.step(epoch=epoch)

            # evaluate
            e = evaluate(self.model, valid_loader, self.device)
            self.ap.append(_calculate_ap(e))
            self.ap_iou_point_5.append(_calculate_ap(e, iou_threshold_idx=0))


def entrance(save_model_path='/mnt/chjinche/test_data/detection/init_model',
             model_type='fasterrcnn_resnet50_fpn',
             pretrained=True):
    model_config = {
        'model_class': 'FasterRCNN',
        'model_type': model_type,
        'pretrained': pretrained
    }
    logger.info('Dump untrained model.')
    logger.info(f'Model config: {model_config}.')
    dumper = pickle_dumper(model_config, 'model_config.pkl')
    save_model_to_directory(save_model_path, dumper)
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(entrance)
