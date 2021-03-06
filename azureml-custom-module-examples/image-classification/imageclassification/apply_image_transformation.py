import fire
from torchvision import transforms
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.transformation_directory import ImageTransformationDirectory


class ApplyImageTransformation:
    # Follow webservice api contract
    def __init__(self):
        self.unloader = transforms.ToPILImage()

    def onstart(self, ports={}, params={}):
        mode = params.get('Mode', None)
        input_transform_dir = ports.get('Input image transformation', None)
        self.transform = self.get_transforms(input_transform_dir, mode)
        logger.info(f'{mode}, transforms {self.transform}.')

        logger.info("Transformation init finished.")

    def get_transforms(self, input_transform, mode):
        if mode == 'For training':
            return input_transform.torch_transform
        if mode == 'For testing':
            raw_transforms = input_transform.transforms
            test_transforms = [
                t for t in raw_transforms if not t[0].startswith('Random')
            ]
            return ImageTransformationDirectory.get_torch_transform(
                test_transforms)
        else:
            # Will never throw this error thanks to UI constraints
            raise TypeError(f"Unsupported transform_type type {mode}.")

    # Follow webservice api contract
    def apply(self, ports={}, params={}):
        input_image_dir = ports.get('Input image directory', None)
        logger.info(f'Applying transform:')
        transformed_dir = input_image_dir.apply_to_images(
            transform=lambda image: self.unloader(
                self.transform(image).squeeze(0)))
        return transformed_dir


def entrance(mode,
             input_transform_path='/mnt/chjinche/test_data/init_transform/',
             input_image_path='/mnt/chjinche/test_data/image_dir_test/',
             output_path='/mnt/chjinche/test_data/transform_test/'):
    params = {'Mode': mode}
    ports = {
        'Input image transformation':
        ImageTransformationDirectory.load(input_transform_path),
        'Input image directory':
        ImageDirectory.load(input_image_path)
    }
    task = ApplyImageTransformation()
    task.onstart(ports=ports, params=params)
    task.apply(ports=ports, params=params).dump(output_path)
    logger.info("Transformed dir dumped")


if __name__ == '__main__':
    fire.Fire(entrance)
