import fire
from pathlib import Path
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import ImageDirectory


def entrance(input_path='/mnt/chjinche/test_data/classification/compressed/image_dataset.zip',
             output_path='/mnt/chjinche/test_data/test/image_dir/'):
    logger.info('Start!')
    logger.info(f'input path {input_path}')
    # TODO:Case 1: input path is torchvision ImageFolder
    # loader_dir = FolderBasedImageDirectory.load_organized(input_path)
    # Case 2: input path is compressed file
    compressed_extensions = {'.tar', '.zip'}
    compressed_path = None
    if Path(input_path).is_file():
        compressed_path = input_path
    else:
        for path in Path(input_path).glob(r'**/*'):
            print(path)
            if path.suffix in compressed_extensions:
                compressed_path = path

    logger.info(f'compressed file path {compressed_path}')
    loader_dir = ImageDirectory.load_compressed(compressed_path)
    # TODO: Case 3: input path is custom directory
    loader_dir.dump(output_path)
    logger.info('Finished.')


if __name__ == '__main__':
    fire.Fire(entrance)
