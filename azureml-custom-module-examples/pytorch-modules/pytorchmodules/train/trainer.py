import time
import torch
from azureml.studio.core.logger import logger
from .trainer_utils import AverageMeter, evaluate


class ClassificationTrainer:
    def __init__(self, model):
        self.model = model

    def train_one_epoch(self, loader, optimizer, epoch, epochs, print_freq=1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        error = AverageMeter()
        # Model on train mode
        # print(self.model)
        self.model.train()
        end = time.time()
        batches = len(loader)
        for batch_idx, (input, target) in enumerate(loader):
            # Create variables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            # Compute output
            output = self.model(input)
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

    def fit(self,
            train_set=None,
            valid_set=None,
            epochs=None,
            batch_size=None,
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
            self.model = self.model.cuda()

        logger.info(f'device count: {torch.cuda.device_count()}')
        if torch.cuda.device_count() > 1:
            # self.model = torch.nn.parallel.DistributedDataParallel(self.model).cuda()
            self.model = torch.nn.DataParallel(self.model).cuda()

        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    nesterov=True,
                                    weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1)
        logger.info('Start training epochs.')
        best_error = 1
        counter = 0
        last_epoch_valid_loss = -1
        best_model = None
        for epoch in range(epochs):
            scheduler.step(epoch=epoch)
            _, train_loss, train_error = self.train_one_epoch(
                loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                epochs=epochs)
            _, valid_loss, valid_error, _ = evaluate(model=self.model,
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
                best_model = self.model

            early_stop = True if counter >= patience else False
            if early_stop:
                logger.info("Early stopped.")
                break
        return best_model
