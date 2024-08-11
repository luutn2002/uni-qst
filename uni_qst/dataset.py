import torch
from qutip.wigner import qfunc
from uni_qst.state_ops import cat, num, thermal_dm, binomial, coherent_dm, fock_dm, gkp
from uni_qst.noise_ops import RandomNoiseApply
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import os

xgrid = 32
ygrid = 32

xvec = np.linspace(-5, 5, xgrid)
yvec = np.linspace(-5, 5, ygrid)

class DatasetGenerator():
    '''
    Data generator for training and testing
    - Methods:
        - state_gen(): generate a single state.
        - generate(): generate a whole dataset.
    '''
    def __init__(self):
        self.label_map = {'fock_dm': 0,
                          'coherent_dm':1,
                          'thermal_dm':2,
                          'num':3,
                          'binomial':4,
                          'cat':5,
                          'gkp':6}
        
        self.function_list = [fock_dm, coherent_dm, thermal_dm, 
                              num, binomial, cat, gkp]
        
    def _generate_single_state(self,
                               apply_noise,
                               to_numpy):
        
        func = np.random.choice(self.function_list)
        label = self.label_map.get(func.__name__)
        if label == None:
            raise Exception(f'Label return None from {func.__name__}, cannnot determine function.')
            
        state, ft = func(32)
        values = state.copy()
        
        if apply_noise: state = RandomNoiseApply(state, to_numpy=to_numpy)
        else: state = torch.from_numpy(qfunc(state, xvec, yvec, g=2))
        label = torch.Tensor([label])
        ft = torch.Tensor([ft])
        values = torch.from_numpy(values.full())
        
        if to_numpy:
            label = label.numpy()
            values = values.numpy()
            ft = ft.numpy()
                    
        return state, label, values, ft
    
    def _generate_multiple_states(self,
                                  num_samples,
                                  apply_noise,
                                  to_numpy):
        
        state_list = None
        label_list = None
        values_list = None
        ft_list = None
        
        for _ in tqdm(np.arange(num_samples)):
            func = np.random.choice(self.function_list)
            label = self.label_map.get(func.__name__)
            if label == None:
                raise Exception(f'Label return None from {func.__name__}, cannnot determine function.')
            
            state, ft = func(32)
            values = state.copy()
            
            if apply_noise: state = RandomNoiseApply(state, to_numpy=to_numpy)
            else: state = torch.from_numpy(qfunc(state, xvec, yvec, g=2))
            values = torch.tensor(values.full(), dtype=torch.complex128)
            # print(ft)
            ft = torch.tensor(ft, dtype=torch.complex128)
                
            if state_list is None: state_list = torch.unsqueeze(state, dim=0)
            else: state_list = torch.cat((state_list, torch.unsqueeze(state, dim=0)), 0)
                                            
            if label_list is None: label_list = [label]
            else: label_list.append(label)
                
            if values_list is None: values_list = torch.unsqueeze(values, dim=0)
            else: values_list = torch.cat((values_list, torch.unsqueeze(values, dim=0)), 0)
                
            if ft_list is None: ft_list = torch.unsqueeze(ft, dim=0)
            else: ft_list = torch.cat((ft_list, torch.unsqueeze(ft, dim=0)), 0)
                            
        label_list = torch.Tensor(label_list)

        if to_numpy:
            state_list = state_list.numpy()
            label_list = label_list.numpy()
            values_list = values_list.numpy()
            ft_list = ft_list.numpy()
        else:
            state_list = torch.unsqueeze(state_list, dim=1)
            label_list = label_list.type(torch.LongTensor)
            values_list = torch.unsqueeze(values_list, dim=1)
            ft_list = torch.unsqueeze(ft_list, dim=1)
            
        return state_list, label_list, values_list, ft_list
    
    def _state_gen(self,
                  random_mode=True,
                  apply_noise=True,
                  to_numpy=True,
                  num_samples=1):
        
        '''
        - Args:
            - state(`string`): determine output state type (choose from label_map or function_list), no effect if random_mode set to True.
            - random_mode(`bool`): if set True, all state generated is randomly selected from function list. If False, all state will be determined by state variable.
            - num_samples(`int`): number of output samples, stored in a numpy array by default.
         
        - Return: 
            - state(torch tensor), label(int), regression label(torch 1D tensor).
        '''
        #Random mode
        if random_mode:
            if num_samples <= 1: return self._generate_single_state(apply_noise, to_numpy)
            
            else: return self._generate_multiple_states(num_samples, apply_noise, to_numpy)
            
    def generate(self,
                dataset_size=10,
                train_split=0.8,
                to_numpy = True,
                save_file = True,
                apply_noise = False,
                train_dir = './train',
                test_dir='./test'):
        
        if to_numpy: file_type = '.npy'
        else: file_type = '.pt'
                    
        train_data_file_dir=train_dir + '/data' + file_type
        train_label_file_dir=train_dir + '/label' + file_type
        train_values_file_dir=train_dir + '/values' + file_type
        train_ft_file_dir=train_dir + '/ft' + file_type
        
        test_data_file_dir=test_dir + '/data' + file_type
        test_label_file_dir=test_dir + '/label' + file_type
        test_values_file_dir=test_dir + '/values' + file_type
        test_ft_file_dir=test_dir + '/ft' + file_type

        train_size = round(dataset_size*train_split)
        test_size = dataset_size-train_size
        
        train_data, train_label, train_value, train_ft = self._state_gen(to_numpy=to_numpy,
                                                                        apply_noise=apply_noise,
                                                                        num_samples=train_size)
        
        test_data, test_label, test_value, test_ft = self._state_gen(to_numpy=to_numpy,
                                                                    apply_noise=apply_noise,
                                                                    num_samples=test_size)
        if save_file:
            if not os.path.exists(train_dir): os.makedirs(train_dir)
            if not os.path.exists(test_dir): os.makedirs(test_dir)
            if to_numpy: 
                np.save(train_data_file_dir, train_data)
                np.save(train_label_file_dir, train_label)
                np.save(train_values_file_dir, train_value)
                np.save(train_ft_file_dir, train_ft)
                np.save(test_data_file_dir, test_data)
                np.save(test_label_file_dir, test_label)
                np.save(test_values_file_dir, test_value)
                np.save(test_ft_file_dir, test_ft)

            else: 
                torch.save(train_data, train_data_file_dir)
                torch.save(train_label, train_label_file_dir)
                torch.save(train_value, train_values_file_dir)
                torch.save(train_ft, train_ft_file_dir)
                torch.save(test_data, test_data_file_dir)
                torch.save(test_label, test_label_file_dir)
                torch.save(test_value, test_values_file_dir)
                torch.save(test_ft, test_ft_file_dir)
                
        else:
            return train_data, train_label, train_value, train_ft, test_data, test_label, test_value, test_ft
        
class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms. For Pytorch compatible model like RFB-Net.
    """
    def __init__(self, 
                 tensors):
        
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
 
    def __getitem__(self, index):
            return self.tensors[0][index], self.tensors[1][index], self.tensors[2][index], self.tensors[3][index]

    def __len__(self):
        return self.tensors[0].size(0)