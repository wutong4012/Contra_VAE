import os
import glob
import random
import datasets
import torch.utils.data as data

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class ContraVAEDataset(data.Dataset):

    def __init__(self, train, config):
        """
        self.data: list[dict]
        dict{T, t, 0, total_t, z0_encode, zt_encode, zT_encode, 
            z0_decode, zt_decode, zT_decode}
        """ 
        super().__init__()
        self.config = config
        
        if train:
            print("Load training data from {}.".format(config.data_params.train_data_path))
            self.data = self._load_data(config.data_params.train_data_path)
            print("There are {} Train samples in all!\n".format(len(self.data)))
            
        else:
            self.data = []
            print("Load testing data from {}.".format(config.data_params.test_data_path))
            pre_data = datasets.load_from_disk(config.data_params.test_data_path)
            for idx in tqdm(range(pre_data.num_rows), desc="Concat_Data"):
                self.data.extend(pre_data[idx]['tokenized_samples'])
            print("Examples: {}".format(self.data[0]))
            print("There are {} test samples in all!\n".format(len(self.data)))

    def _load_data(self, path):
        cache_dict_paths = []
        random_list = random.sample(range(200), self.config.data_params.file_nums)  # different results every time
        all_paths = glob.glob(os.path.join(path, '*'))
        for idx in random_list:
            cache_dict_paths.append(all_paths[idx])
        train_ds, res = [], []
        p = ProcessPoolExecutor(max_workers=42)
        
        for path in tqdm(cache_dict_paths, desc="Submit_Process"):
            res.append(p.submit(datasets.load_from_disk, path))
        p.shutdown(wait=True)
        for future in tqdm(res, desc="Load_Dataset"):
            train_ds.append(future.result())
        train_ds = datasets.concatenate_datasets(train_ds)
        
        train_data = []
        if self.config.exp_params.just_test:
            data_num = 100
        else:
            data_num = train_ds.num_rows
        for idx in tqdm(range(data_num), desc="Concat_Data"):
            train_data.extend(train_ds[idx]['tokenized_samples'])
            
        random_list = random.sample(range(len(train_data)), 3)
        for i in random_list:
            print("Examples: {}".format(train_data[i]))
        
        return train_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data) - 1
