import json
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast
import torch
from data.const import task_rel_labels, task_ner_labels, ENTITY_PADDING, RELATION_PADDING




def _read(json_file):
    # 读取的时候可以一次读取两种文件
    # 仅仅是将json文件转化为list 每个list里面属于字典

    gold_docs = [line for line in json.load(open(json_file))]

    return gold_docs


def _analyze(json_file):
    max_e = 0
    sum_e = 0.0
    max_r = 0
    sum_r = 0.0
    count = 0
    dic_e = {}
    dic_r = {}
    for line in json.load(open(json_file)):
        count += 1
        sum_r += len(line['entities'])
        sum_e += len(line['entities'])
        if len(line['entities']) not in dic_e:
            dic_e[len(line['entities'])] = 1
        else:
            dic_e[len(line['entities'])] += 1
        if len(line['relations']) not in dic_r:
            dic_r[len(line['relations'])] = 1
        else:
            dic_r[len(line['relations'])] += 1
        if len(line['entities']) > max_e:
            max_e = len(line['entities'])
        if len(line['relations']) > max_r:
            max_r = len(line['relations'])
    x_e = []
    y_e = []
    x_r = []
    y_r = []
    for e in sorted(dic_e.keys()):
        x_e.append(e)
        y_e.append(dic_e[e])
    for r in sorted(dic_r.keys()):
        x_r.append(r)
        y_r.append(dic_r[r])
    plt.plot(x_e, y_e)
    plt.show()
    plt.plot(x_r, y_r)
    plt.show()
    e = sum_e / count
    r = sum_r / count


class Dataset:
    def __init__(self, args, file):
        self.tokenizer = BertTokenizerFast.from_pretrained(args.PRETRAINED_MODEL_NAME)
        self.args = args
        self.js = _read(file)
        self.re_label2id, self.re_id2label, self.re_num_labels, \
        self.en_label2id, self.en_id2label, self.en_num_labels = self._label2id()
        self._get_input_tensor()
        self._entity_padding()
        self._relation_padding()


    def __getitem__(self, ix):
        return self.js[ix]

    def __len__(self):
        return len(self.js)

    def _entity_padding(self):
        for i, sample in enumerate(self.js):
            for j, entity in enumerate(sample['entities']):
                self.js[i]['entities'][j][0] = sample['start2id'][entity[0]].numpy().tolist()
                self.js[i]['entities'][j][1] = sample['end2id'][entity[1]].numpy().tolist()
                self.js[i]['entities'][j][3] = self.en_label2id[self.js[i]['entities'][j][3]]
                del self.js[i]['entities'][j][2]
            if len(self.js[i]['entities']) < self.args.ENTITY_NUM:
                for t in range(self.args.ENTITY_NUM - len(self.js[i]['entities'])):
                    self.js[i]['entities'].append(ENTITY_PADDING)
            else:
                self.js[i]['entities'] = self.js[i]['entities'][:self.args.ENTITY_NUM]
            entities = self.js[i]['entities']
            self.js[i]['entities'] = torch.tensor(entities, dtype=torch.long).cuda()

    def _relation_padding(self):
        for i, sample in enumerate(self.js):
            for j, relation in enumerate(sample['relations']):
                self.js[i]['relations'][j][2] = self.re_label2id[self.js[i]['relations'][j][2]]
            # 将relation加入cuda
            if len(self.js[i]['relations']) < self.args.RELATION_NUM:
                for t in range(self.args.RELATION_NUM - len(self.js[i]['relations'])):
                    self.js[i]['relations'].append(RELATION_PADDING)
            else:
                self.js[i]['relations'] = self.js[i]['relations'][:self.args.RELATION_NUM]
            relations = self.js[i]['relations']
            self.js[i]['relations'] = torch.tensor(relations, dtype=torch.long).cuda()


    def _get_input_tensor(self):
        for i, sample in enumerate(self.js):
            # start2id和end2id记录每个word在tokenizer之后的位置
            text = sample['text']
            start2id = []
            end2id = []
            tokens = [self.tokenizer.cls_token]
            for token in text:
                start2id.append(len(tokens))
                sub_token = self.tokenizer.tokenize(token)
                tokens += sub_token
                end2id.append(len(tokens) - 1)
            tokens.append(self.tokenizer.sep_token)
            mask = [1]*len(tokens)
            if len(tokens) < self.args.MAX_LEN:
                for t in range(self.args.MAX_LEN - len(tokens)):
                    tokens.append(self.tokenizer.pad_token)
                    mask.append(0)
            if len(start2id) < self.args.MAX_LEN:
                for t in range(self.args.MAX_LEN - len(start2id)):
                    start2id.append(-1)
            if len(end2id) < self.args.MAX_LEN:
                for t in range(self.args.MAX_LEN - len(end2id)):
                    end2id.append(-1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            self.js[i]['input_ids'] = torch.tensor(input_ids, dtype=torch.long).cuda()
            self.js[i]['mask'] = torch.tensor(mask, dtype=torch.bool).cuda()
            self.js[i]['text'] = tokens
            # 之后要用先不移动到cuda
            self.js[i]['start2id'] = torch.tensor(start2id, dtype=torch.long)
            self.js[i]['end2id'] = torch.tensor(end2id, dtype=torch.long)

    def _label2id(self):
        re_label_list = [self.args.negative_label] + task_rel_labels[self.args.task]
        re_label2id = {label: i for i, label in enumerate(re_label_list)}
        re_id2label = {i: label for i, label in enumerate(re_label_list)}
        re_num_labels = len(re_label_list)

        en_label_list = task_ner_labels[self.args.task]
        en_label2id = {label: i for i, label in enumerate(en_label_list)}
        en_id2label = {i: label for i, label in enumerate(en_label_list)}
        en_num_labels = len(en_label_list)
        return re_label2id, re_id2label, re_num_labels, en_label2id, en_id2label, en_num_labels





# def collate_fn(batch):
#     #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
#     text = []
#
#     batch = list(zip(batch))
#     labels = torch.tensor(batch[0], dtype=torch.int32)
#     texts = batch[1]
#     del batch
#     return labels, texts
