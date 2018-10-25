import torch
import torch.nn as nn
from hyperparameter import *
import numpy as np
class myEmbedding(nn.Module):
    def __init__(self, vocab_dim, input):
        super(myEmbedding,self).__init__()
        self.hyperparameter = Hyperparameter()
        self.embedding = nn.Embedding(vocab_dim, self.hyperparameter.class_num, scale_grad_by_freq=True, sparse=True)
        self.forward(input)

    def forward(self, input):
        output_list = []
        for i in input:
            output = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            for j in i.word_index:
                output += np.array(self.embedding[j])
            output_list.append(self.softmax((1/self.hyperparameter.batch_size)*output))
        return output_list

    def softmax(self, result):
        result_list = []
        bottom = 0
        max_idx = self.get_max_index(result)
        for index, value in enumerate(result):
            bottom += np.exp(value - result[max_idx])
        for index, value in enumerate(result):
            result_list.append(np.exp(value - result[max_idx])/bottom)
        return result_list

    def get_max_index(self, result):
        max, index = result[0],0
        for idx in range(len(result)):
            if result[idx] > max:
                max, index = result[idx], idx
        return index

