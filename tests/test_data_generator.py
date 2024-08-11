from uni_qst.dataset import DatasetGenerator
import os

def test_data_generator(train_dir='./train', 
                       test_dir='./test',
                       to_numpy=True):
    ext = '.npy' if to_numpy else '.pt'
    generator = DatasetGenerator()
    generator.generate(train_dir=train_dir, test_dir=test_dir)
    assert (os.path.isfile(os.path.join(train_dir, f'data{ext}')) and \
            os.path.isfile(os.path.join(train_dir, f'label{ext}')) and \
            os.path.isfile(os.path.join(train_dir, f'values{ext}')) and \
            os.path.isfile(os.path.join(train_dir, f'ft{ext}')) and \
            os.path.isfile(os.path.join(test_dir, f'data{ext}')) and \
            os.path.isfile(os.path.join(test_dir, f'label{ext}')) and \
            os.path.isfile(os.path.join(test_dir, f'values{ext}')) and \
            os.path.isfile(os.path.join(test_dir, f'ft{ext}')))