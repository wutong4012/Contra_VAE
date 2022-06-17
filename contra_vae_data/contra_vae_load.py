import re
import os
import copy
import random

from concurrent.futures import ProcessPoolExecutor
from contra_vae.pytorch_transformers import BertTokenizer

_BERT_TOKENIZERS = '/cognitive_comp/wutong/source/model_base/bert-base/'

# 缓存文件
_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/wutong/source/data_base/contra_vae/train_split_exp1'
_CACHE_TEST_DATA_PATH = '/cognitive_comp/wutong/source/data_base/contra_vae/test_exp1'

_MAX_SENTENCE_LENGTH = 50
_NUM_PROC = 128
_TRAIN_SHARD_PART = 200


def get_dataset(doc_info):
    doc_samples = []
    for idx in range(2, len(doc_info)):
        sentence_info = doc_info[idx]
        sentence_id = sentence_info['sentence_id']

        T = sentence_id
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t
        assert t1 < t2 and t2 < T
        
        y_0 = doc_info[idx - T + t1]['sentence']
        y_t = doc_info[idx - T + t2]['sentence']
        y_T = doc_info[idx]['sentence']
        t_ = t1
        t = t2
        total_doc = sentence_info['total_doc_sentences']
        
        sample = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        doc_samples.append(sample)
    
    return doc_samples


def preprocess(text:str):
    doc_info = []
    split_sentence = re.split(r',|;|!|\?|\.|，|。|；|！|？|…', text) # 英文?和.需要转义符\
    for idx in range(len(split_sentence)-1, -1, -1): # reversed order.
        # delete sentences without Chinese.
        if re.compile(u'[\u4e00-\u9fa5]+').search(split_sentence[idx]) == None or len(split_sentence[idx]) <= 5:
            del split_sentence[idx]
    if len(split_sentence) >= 3:
        for idx in range(len(split_sentence)):
            sentence_info = {
                    "sentence": split_sentence[idx][:_MAX_SENTENCE_LENGTH],
                    "sentence_id": idx, # start at index 0.
                    "total_doc_sentences": len(split_sentence) - 1,
                }
            doc_info.append(sentence_info)
    doc_samples = get_dataset(doc_info)
    
    return doc_samples


def tokenize_data(doc_sample, tokenizers):
    tokenized_samples = []
    for sample in doc_sample:
        z0_encode = tokenizers[0].encode(sample['y_0'], add_special_tokens=True)
        zt_encode = tokenizers[0].encode(sample['y_t'], add_special_tokens=True)
        zT_encode = tokenizers[0].encode(sample['y_T'], add_special_tokens=True)
        
        z0_decode = tokenizers[1].encode('<BOS>' + sample['y_0'] + '<EOS>')
        zt_decode = tokenizers[1].encode('<BOS>' + sample['y_t'] + '<EOS>')
        zT_decode = tokenizers[1].encode('<BOS>' + sample['y_T'] + '<EOS>')
        
        per_sample = {
            'z0_encode': str(z0_encode),
            'zt_encode': str(zt_encode),
            'zT_encode': str(zT_encode),
            'z0_decode': str(z0_decode),
            'zt_decode': str(zt_decode),
            'zT_decode': str(zT_decode),
            't_': sample['t_'],
            't': sample['t'],
            'T': sample['T'],
            'total_t': sample['total_t'],
        }
        tokenized_samples.append(per_sample)

    return tokenized_samples


def _generate_cache_arrow(index, ds):
    print('saving dataset shard {}'.format(index))
    ds.save_to_disk(os.path.join(_CACHE_TRAIN_DATA_PATH, 'part_{}'.format(index)))
    return 'saving dataset shard {} done'.format(index)


def generate_arrow_cache(num_proc=1) -> None:
    '''
    读取wudao_180g原始数据，并进行预处理，之后tokenize生成datasets
    同时利用seed 42做shuffle 缓存下来
    '''
    import sys
    sys.path.append('../../')
    from fs_datasets import load_dataset
    ds = load_dataset('wudao_180g', num_proc=200)
    ds = ds['train'].train_test_split(train_size=0.99, test_size=0.01, seed=42)
    ds = ds['test'].train_test_split(train_size=0.99, test_size=0.01, seed=42)
    print(ds)
    encoder_tokenizer = BertTokenizer.from_pretrained(_BERT_TOKENIZERS)
    decoder_tokenizer = copy.deepcopy(encoder_tokenizer)
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    decoder_tokenizer.add_special_tokens(special_tokens_dict)

    def _tokenizer(example):
        doc_samples = preprocess(example['text'])
        # if len(doc_samples) > 5:
        #     random_list = random.sample(range(len(doc_samples)), 10)
        #     for i in random_list:
        #         print("Examples: {}".format(doc_samples[i]))

        tokenized_samples = tokenize_data(doc_samples, [encoder_tokenizer, decoder_tokenizer])
        # if len(tokenized_samples) > 5:
        #     random_list = random.sample(range(len(tokenized_samples)), 5)
        #     for i in random_list:
        #         print("Examples: {}".format(tokenized_samples[i]['z_0']))
        #         print(tokenizer.convert_ids_to_tokens(eval(tokenized_samples[i]['z_0'])))
        #         print(type(tokenized_samples[i]['z_0']))
        
        return {
            'tokenized_samples': tokenized_samples,
        }

    tokenized_ds = ds.map(
        _tokenizer,
        num_proc=num_proc,
        remove_columns=['text'])

    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []
    train_shard_part = _TRAIN_SHARD_PART
    for i in range(0, train_shard_part):
        res.append(p.submit(_generate_cache_arrow, i,
                            tokenized_ds['train'].shard(train_shard_part, i)))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)

    tokenized_ds['test'].save_to_disk(_CACHE_TEST_DATA_PATH)

    print('done')


if __name__ == '__main__':
    generate_arrow_cache(num_proc=_NUM_PROC)
