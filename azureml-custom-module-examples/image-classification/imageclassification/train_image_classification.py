from pathlib import Path
import time
import fire
import torch
from azureml.designer.model.io import save_pytorch_state_dict_model
from azureml.designer.model.model_spec.task_type import TaskType
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.logger import logger
from azureml.studio.core.io.model_directory import load_model_from_directory, pickle_loader
from .utils import AverageMeter, evaluate
from . import modellib


def train_epoch(model, loader, optimizer, epoch, epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    # Model on train mode
    model.train()
    end = time.time()
    batches = len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        # Create variables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        # Compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)
        # Measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(
            torch.ne(pred.squeeze(), target.cpu()).float().sum().item() /
            batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                f'Epoch: [{epoch + 1}/{epochs}]',
                f'Iter: [{batch_idx + 1}/{batches}]',
                f'Avg_Time_Batch/Avg_Time_Epoch: {batch_time.val:.3f}/{batch_time.avg:.3f}',
                f'Avg_Loss_Batch/Avg_Loss_Epoch: {losses.val:.4f}/{losses.avg:.4f}',
                f'Avg_Error_Batch/Avg_Error_Epoch: {error.val:.4f}/{error.avg:.4f}'
            ])
            logger.info(res)
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model,
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
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(torch.cuda.is_available()),
        num_workers=0)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        num_workers=0)
    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                nesterov=True,
                                weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1)
    logger.info('Start training epochs.')
    best_error = 1
    counter = 0
    for epoch in range(epochs):
        _, train_loss, train_error = train_epoch(model=model,
                                                 loader=train_loader,
                                                 optimizer=optimizer,
                                                 epoch=epoch,
                                                 epochs=epochs)
        scheduler.step()
        _, valid_loss, valid_error, _ = evaluate(model=model,
                                                 loader=valid_loader)
        # Determine if model is the best
        if valid_error < best_error:
            is_best = True
            best_error = valid_error
        else:
            is_best = False

        # Early stop
        if epoch == 0:
            last_epoch_valid_loss = valid_loss
        else:
            if valid_loss >= last_epoch_valid_loss:
                counter += 1
            else:
                counter = 0
            last_epoch_valid_loss = valid_loss

        logger.info(
            f'valid loss did not decrease consecutively for {counter} epoch')
        # TODO: save checkpoint files, but removed now to increase web service deployment efficiency.
        logger.info(','.join([
            f'Epoch {epoch + 1:d}', f'train_loss {train_loss:.6f}',
            f'train_error {train_error:.6f}', f'valid_loss {valid_loss:.5f}',
            f'valid_error {valid_error:.5f}'
        ]))
        if is_best:
            logger.info(
                # f'Get better top1 accuracy: {1-best_error:.4f} will saving weights to {best_checkpoint_name}'
                f'Get better top1 accuracy: {1-best_error:.4f}, best checkpoint will be updated.'
            )

        early_stop = True if counter >= patience else False
        if early_stop:
            logger.info("Early stopped.")
            break

    return model


def entrance(input_model_path='../init_model',
             train_data_path='../transform_train/',
             valid_data_path='../transform_test/',
             save_model_path='../saved_model',
             epochs=20,
             batch_size=16,
             learning_rate=0.001,
             random_seed=231,
             patience=2):
    logger.info("Start training.")
    logger.info(f"data path: {train_data_path}")
    logger.info(f"data path: {valid_data_path}")
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
    model = train(model=model,
                  train_set=train_set,
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
    save_pytorch_state_dict_model(model,
                                  init_params=model_config,
                                  path=save_model_path,
                                  task_type=TaskType.MultiClassification,
                                  label_map=id_to_class_dict,
                                  conda=conda)
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
