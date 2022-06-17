import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch_data, pad_token_id):
    t_, t, T, total_t, batch_tensor = [], [], [], [], {}
    z0_encode_input_ids, zt_encode_input_ids, zT_encode_input_ids = [], [], []
    z0_decode_input_ids, zt_decode_input_ids, zT_decode_input_ids = [], [], []

    for per_data in batch_data:
        t_.append(per_data['t_'])
        t.append(per_data['t'])
        T.append(per_data['T'])
        total_t.append(per_data['total_t'])

        z0_encode = eval(per_data['z0_encode'])
        zt_encode = eval(per_data['zt_encode'])
        zT_encode = eval(per_data['zT_encode'])
        
        z0_decode = eval(per_data['z0_decode'])
        zt_decode = eval(per_data['zt_decode'])
        zT_decode = eval(per_data['zT_decode'])
        
        z0_encode_input_ids.append(z0_encode)
        zt_encode_input_ids.append(zt_encode)
        zT_encode_input_ids.append(zT_encode)
        
        z0_decode_input_ids.append(z0_decode)
        zt_decode_input_ids.append(zt_decode)
        zT_decode_input_ids.append(zT_decode)
    
    z0_encode_input_ids = pad_sequence(
        [torch.tensor(x) for x in z0_encode_input_ids], batch_first=True, padding_value=pad_token_id)
    zt_encode_input_ids = pad_sequence(
        [torch.tensor(x) for x in zt_encode_input_ids], batch_first=True, padding_value=pad_token_id)
    zT_encode_input_ids = pad_sequence(
        [torch.tensor(x) for x in zT_encode_input_ids], batch_first=True, padding_value=pad_token_id)
    
    z0_decode_input_ids = pad_sequence(
        [torch.tensor(x) for x in z0_decode_input_ids], batch_first=True, padding_value=pad_token_id)
    zt_decode_input_ids = pad_sequence(
        [torch.tensor(x) for x in zt_decode_input_ids], batch_first=True, padding_value=pad_token_id)
    zT_decode_input_ids = pad_sequence(
        [torch.tensor(x) for x in zT_decode_input_ids], batch_first=True, padding_value=pad_token_id)
    
    batch_tensor['z0_encode'], batch_tensor['zt_encode'], batch_tensor['zT_encode'] = \
        z0_encode_input_ids, zt_encode_input_ids, zT_encode_input_ids
    batch_tensor['z0_decode'], batch_tensor['zt_decode'], batch_tensor['zT_decode'] = \
        z0_decode_input_ids, zt_decode_input_ids, zT_decode_input_ids
    batch_tensor['t_'], batch_tensor['t'], batch_tensor['T'], batch_tensor['total_t'] = \
        torch.tensor(t_), torch.tensor(t), torch.tensor(T), torch.tensor(total_t)

    return batch_tensor


def create_dataloader(dataset, config, pad_token_id, batch_size, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
        num_workers=config.data_params.dataloader_workers,
    )

    return loader
