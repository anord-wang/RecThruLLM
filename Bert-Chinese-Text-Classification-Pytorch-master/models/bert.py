# coding: UTF-8
import torch
import torch.nn as nn
import re
import os
import pickle
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'

        # self.train_path = dataset + '/data/train.txt'                                # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
            # dataset + '/data/class.txt').readlines()]                                # 类别名单
        
        # 数据集路径
        server_root = '/home/local/ASURITE/xwang735/LLM4REC/LLM4RecAgent'                                          
        data_root = os.path.join(server_root, 'dataset', dataset)
        score_prediction_data_path = os.path.join(data_root, 'score_prediction.txt')
        contrast_learning_data_path = os.path.join(data_root, 'score_prediction.txt')
        self.data_path = score_prediction_data_path
        self.train_path = os.path.join(data_root, 'score_prediction_train.txt')
        self.dev_path = os.path.join(data_root, 'score_prediction_val.txt')
        self.test_path = os.path.join(data_root, 'score_prediction_test.txt')
        # self.data_path = contrast_learning_data_path
        self.class_list = [x.strip() for x in open(data_root + '/score_prediction_class.txt').readlines()]

        # 模型训练结果
        ori_model_root = os.path.join(server_root, 'model', dataset, 'rec')
        model_root = os.path.join(server_root, 'model', dataset, 'score_prediction')
        self.model_save_path = os.path.join(model_root, self.model_name + '.ckpt')
        self.embedding_save_path = model_root
        self.user_emb_load_path = os.path.join(ori_model_root, f"user_embeddings_1.pt")
        self.item_emb_load_path = os.path.join(ori_model_root, f"item_embeddings_1.pt")
        self.user_emb_save_path = os.path.join(model_root, f"user_embeddings_1.pt")
        self.item_emb_save_path = os.path.join(model_root, f"item_embeddings_1.pt")


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.hidden_size = 768
        self.vocab_size = len(self.tokenizer.vocab)

        self.initializer_range = 0.02

        meta_path = os.path.join(data_root, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)
        self.num_users = meta_data["num_users"]
        self.num_items = meta_data["num_items"]
        self.user_token_encoder = self._add_user_token_encoder()
        self.item_token_encoder = self._add_item_token_encoder()
    
    def _add_user_token_encoder(self):
        return {"user_{}".format(i):(i+self.vocab_size) 
                for i in range(self.num_users)}
    
    def _add_item_token_encoder(self):
        return {"item_{}".format(j):(j+self.vocab_size+self.num_users)
                for j in range(self.num_items)}

    def modified_tokenizer(self, text):
        '''
            In this function, we break down the sentence that 
            describes user/item features or their historical 
            interactions into pieces, where the ID word like
            user_i or item_j is kept as a single piece. 

            E.g.,
                text = "This is user_1's comment about item_3 
                        after he bought the item"
                pieces = ['This is', 'user_1', "'s comment about", 
                        'item_3', ' after he bought the item']

            Note that we keep the space on the left of a word to 
            show that the word does not appear on the beginning 
            part of a sentence.
        '''
        pattern = r'(user_\d+|item_\d+)'
        matches = re.findall(pattern, text)
        pieces = re.split(pattern, text)
        pieces = [piece.rstrip() for piece in pieces if piece.rstrip()]
        split_tokens = []
        # pieces = self._pre_tokenize(text)
        for piece in pieces:
            # If piece is a user ID
            # piece is itself a token
            if piece in self.user_token_encoder.keys():
                split_tokens.append(piece)
            # If piece is an item ID
            # piece is also a token
            elif piece in self.item_token_encoder.keys():
                split_tokens.append(piece)
            # If piece is a sentence
            # Use the original tokenization to
            # further break down piece
            else:
                split_tokens += self.tokenizer.tokenize(piece)
        return split_tokens
    
    def modified_convert_tokens_to_ids(self, tokens):
        '''
            This function converts a list of tokens into a list of token IDs.
            The difference between this function and the original one is that
            this function uses the modified tokenization function to tokenize
            the input tokens.
        '''
        self.tokenizer.vocab.update(self.user_token_encoder)
        self.tokenizer.vocab.update(self.item_token_encoder)
        return self.tokenizer.convert_tokens_to_ids(tokens)
    

        
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.hidden_size)
        self.item_embeddings = nn.Embedding(self.num_items, config.hidden_size)

        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long()
        #print('vocab_mask: ', vocab_mask)
        user_mask = ((input_ids >= self.vocab_size) & (input_ids < self.vocab_size + self.num_users)).long()
        #print('user_mask: ', vocab_mask)
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()
        #print('item_mask: ', vocab_mask)

        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size - 1)
        vocab_embeddings = self.bert.embeddings.word_embeddings(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)

        # IDs outside of user range are set to 0
        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users - 1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)

        # IDs outside of item range are set to 0
        item_ids = ((input_ids - self.vocab_size - self.num_users) * item_mask).clamp_(0, self.num_items - 1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + item_embeddings
        return input_embeddings

    # def forward(self, x):
    def forward(self, inputs=None):
        # context = x[0]  # 输入的句子
        # mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # out = self.fc(pooled)

        input_ids = inputs[0]
        seq_len = inputs[1]
        mask = inputs[2]

        input_embeddings = self.embed(input_ids)
        outputs = self.bert(inputs_embeds=input_embeddings, attention_mask=mask, return_dict=True)
        pooled_output = outputs.pooler_output
        out = self.fc(pooled_output)
        return out
