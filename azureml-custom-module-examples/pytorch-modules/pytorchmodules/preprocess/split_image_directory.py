import fire
from sklearn.model_selection import train_test_split
from azureml.studio.core.logger import logger
from azureml.studio.core.io.image_directory import ImageDirectory, FolderBasedImageDirectory


def get_stratified_split_list(lst, fraction):
    n = len(lst)
    # TODO: how to stratified split for multi-label scenario
    labels = [d['category_id'] for d in lst]
    train_idx, test_idx, train_label, test_label = train_test_split(
        list(range(n)), labels, stratify=labels, train_size=fraction)
    return train_idx, test_idx


def split_images(src_path, tgt_train_path, tgt_test_path, fraction):
    # TODO: use multi-label stratified split
    # from skmultilearn.model_selection import iterative_train_test_split
    # X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size = 0.5)
    loaded_dir = ImageDirectory.load(src_path)
    lst = loaded_dir.image_lst
    logger.info(f'Start splitting.')
    train_set_idx, test_set_idx = get_stratified_split_list(lst, fraction)
    logger.info(f'Got stratified split list. train {len(train_set_idx)}, test {len(test_set_idx)}.')
    train_set_dir = loaded_dir.get_sub_dir(train_set_idx)
    test_set_dir = loaded_dir.get_sub_dir(test_set_idx)
    logger.info('Dump train set.')
    train_set_dir.dump(tgt_train_path)
    logger.info('Dump test set.')
    test_set_dir.dump(tgt_test_path)


def entrance(src_path='/mnt/chjinche/test_data/image_dir/',
             fraction=0.9,
             tgt_train_path='/mnt/chjinche/test_data/image_dir_train/',
             tgt_test_path='/mnt/chjinche/test_data/image_dir_test/'):
    logger.info('Start!')
    split_images(src_path, tgt_train_path, tgt_test_path, fraction)
    logger.info('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
