from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser
from instence import *
from hyperparameter import Hyperparameter
import numpy as np
import sys
import re
import os
import random
import torch
import time
import nltk.tree
import pickle

torch.manual_seed(666)
random.seed(666)
np.random.seed(666)
class Classifier:
    def __init__(self):
        self.feature_alphabet = feature_alphabet()
        self.all_inst = all_inst()
        self.hyperparameter_1 = Hyperparameter()


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def read_file(self, path):
        Inst_list = []
        parser_sentence = []
        f = open(path, encoding = "UTF-8")
        for line in f.readlines():
            m_1 = inst()
            m_p = inst()
            x = line.strip().split('|||')
            m_1.word = self.clean_str(x[0]).split(' ')
            m_p.word = self.clean_str(x[0])
            # m_1.word = x[0].split(' ')
            # m_p.word = x[0]
            m_1.label = x[1].strip()
            m_p.label = x[1].strip()
            Inst_list.append(m_1)
            parser_sentence.append(m_p)
        f.close()
        return Inst_list, parser_sentence

    def extract_h2_parser(self, sentence):
        list = []
        parser = StanfordParser(model_path="E:/Stanford parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        t = parser.raw_parse(str(sentence))
        for i in t:
            for j in i.subtrees(lambda i: i.height() == 2):
                list.append(str(j))
        return list

    def extract_h3_parser(self, sentence):
        list = []
        parser = StanfordParser(model_path="E:/Stanford parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        t = parser.raw_parse(str(sentence))
        for i in t:
            for j in i.subtrees(lambda i: i.height() == 3):
                list.append(str(j))
        return list

    def extract_h4_parser(self, sentence):
        list = []
        parser = StanfordParser(model_path="E:/Stanford parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        t = parser.raw_parse(str(sentence))
        for i in t:
            for j in i.subtrees(lambda i: i.height() == 4):
                list.append(str(j))
        return list

    def save_depend_tree(self, parser_sentence):
        depend_tree = []
        eng_parser = StanfordDependencyParser(
            model_path="E:/Stanford parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishFactored.ser.gz")
        for i in range(len(parser_sentence)):
            print(i)
            sentence = []
            res = list(eng_parser.parse(str(parser_sentence[i].word).split()))
            for row in res[0].triples():
                sentence.append(row)
            depend_tree.append(sentence)
        pickle.dump(depend_tree, open("test_depend_tree.txt", 'wb'))

    def extract_depend_parser(self, idx):
        feat = []
        dev_depend_tree = pickle.load(open("train_depend_tree.txt", "rb"))
        # print(dev_depend_tree[0])
        row = dev_depend_tree[idx]
        for i in row:
            # print('father = ' + i[0][0] + ' # ' + 'child = ' + i[2][0])
            feat.append('father = ' + i[0][0] + ' # ' + 'child = ' + i[2][0])
        return feat
        # print(str('father = ' + row[0][0] + ' # ' + 'child = ' + row[2][0]))

    # def extract_depend_parser(self, sentence):
    #     parser_feature = []
    #     eng_parser = StanfordDependencyParser(model_path="E:/Stanford parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    #     res = list(eng_parser.parse(str(sentence).split()))
    #
    #     for row in res[0].triples():
    #         parser_feature.append(str('father = ' + row[0][0] + ' # ' + 'child = ' + row[2][0]))
    #     return parser_feature




    def extract_sentence_feature_and_label_encoding(self, Inst_list,path):
        all_inst_feature = []
        dev_depend_tree = pickle.load(open(path, "rb"))
        for i in range(len(Inst_list)):
            # print("提取第{}句话特征".format(i))
            example = Example()
            for idx in range(len(Inst_list[i].word)):
                example.word_index.append('unigram = ' + Inst_list[i].word[idx])
            # example.word_index.extend(self.extract_h2_parser(parser_sentence[i].word))
            for idx in range(len(Inst_list[i].word) - 1):
                example.word_index.append('bigram = ' + Inst_list[i].word[idx] + '#' + Inst_list[i].word[idx + 1])
            # example.word_index.extend(self.extract_h3_parser(parser_sentence[i].word))
            for idx in range(len(Inst_list[i].word) - 2):
                example.word_index.append('trigram = ' + Inst_list[i].word[idx] + '#' + Inst_list[i].word[idx + 1] + '#' + Inst_list[i].word[idx + 2])
            # example.word_index.extend(self.extract_h4_parser(parser_sentence[i].word))

            # for j in dev_depend_tree[i]:
            #     example.word_index.append('father = ' + j[0][0] + '#' + 'child = ' + j[2][0])
            # father = () # child = () 模板

            # for j in dev_depend_tree[i]:
            #     example.word_index.append('father = ' + j[0][0] + ' # ' + 'child = ' + j[2][0] +  ' # ' + 'relation = '+ j[1] )
            #father = () # child = () #ralation = ()模板

            # for j in dev_depend_tree[i]:
            #     example.word_index.append('#' + j[0][0] + '#' + 'pos = ' + j[0][1])
            #     example.word_index.append('#' + j[2][0] + '#' + 'pos = ' + j[2][1])
            # () # pos = () 模板

            # for j in dev_depend_tree[i]:
            #     example.word_index.append('father = ' + j[0][0] + '#' + 'pos=' + j[0][1] + '#' + 'child = ' + j[2][0] + '#' + 'pos=' + j[2][1])

            for j in dev_depend_tree[i]:
                example.word_index.append('father = ' + j[0][0] + '#' + 'pos=' + j[0][1] + '#' + 'child = ' + j[2][0] + '#' + 'pos=' + j[2][1]+ '#' + 'relation='+ j[1])

            # print(example.word_index)
            # example.word_index.extend(self.extract_depend_parser(i))
            # print('加入依存句法特正后',example.word_index)
            if Inst_list[i].label == '0':
                example.label_index = [0, 0, 0, 0, 1]
                example.max_label_index = 4
            elif Inst_list[i].label == '1':
                example.label_index = [0, 0, 0, 1, 0]
                example.max_label_index = 3
            elif Inst_list[i].label == '2':
                example.label_index = [0, 0, 1, 0, 0]
                example.max_label_index = 2
            elif Inst_list[i].label == '3':
                example.label_index = [0, 1, 0, 0, 0]
                example.max_label_index = 1
            elif Inst_list[i].label == '4':
                example.label_index = [1, 0, 0, 0, 0]
                example.max_label_index = 0

            all_inst_feature.append(example)

            # print(example.word_index)
            # if os.path.exists("./train_phrase_feature.txt"):
            #     file = open("./train_phrase_feature.txt", "a")
            # else:
            #     file = open("./train_phrase_feature.txt", "w")
            # file.write(str(example.word_index) + 'label= '+str(example.label_index) + 'max_label_idx= '+ str(example.max_label_index))
            # file.write('\n')
            # file.close()
        # pickle.dump(all_inst_feature, open("dev_depend_feat.txt", 'wb'))
        return all_inst_feature

    def creat_feature_alphabet(self, all_feat):
        a = self.feature_alphabet
        for idx in range(len(all_feat)):
            for word in all_feat[idx].word_index:
                a.list.append(word)
                # if word not in a.list:
                #     a.list.append(word)
        # print(len(a.list))
        # a.list = set(a.list)
        # a.list = list(a.list)

        a.list = self.del_dup(a.list)
        # print(len(a.list))
        # print('====')
        for idx in range(len(a.list)):
            e = a.list[idx]
            a.dict[e] = idx
            # if os.path.exists("./phrase_parser_alphabet.txt"):
            #     file = open("./phrase_parser_alphabet.txt", "a")
            # else:
            #     file = open("./phrase_parser_alphabet.txt", "w")
            # file.write(str(idx) + ' ' + a.list[idx])
            # file.write('\n')
            # file.close()
        # pickle.dump(a, open("pos+_dict.txt", 'wb'))
        return a

        # if os.path.exists("./dev_feature.txt"):
        #     file = open("./dev_feature.txt", "a")
        # else:
        #     file = open("./dev_feature.txt", "w")
        # file.write("dev_feature " + str(a.list) + "\n")
        # file.close()
        # return a
    def del_dup(self, list):
        if list:
            list.sort()
            last = list[-1]
            for i in range(len(list) - 2, -1, -1):
                if last == list[i]:
                    del list[i]
                else:
                    last = list[i]
        return list
    def one_hot_encoding(self, dataset,feat_alphabet,path):
        one_hot_list = []
        all_Inst_feature = self.extract_sentence_feature_and_label_encoding(dataset,path)
        # feat_alphabet = self.creat_feature_alphabet(self.extract_sentence_feature_and_label_encoding(train_Inst))
        # feat_alphabet = pickle.load(open("unclean.txt", "rb"))
        for exam in all_Inst_feature:
            one_hot = Example()
            one_hot.label_index = exam.label_index
            one_hot.max_label_index = exam.max_label_index
            for j in exam.word_index:
                if j in feat_alphabet.dict:
                    one_hot.word_index.append(feat_alphabet.dict[j])
            one_hot_list.append(one_hot)
        return one_hot_list

    def Init_weight_array(self, train_Inst):
        feat_alphabet =self.creat_feature_alphabet(train_Inst)
        self.weight_array = np.random.rand(len(feat_alphabet.list), self.hyperparameter_1.class_num)
        return self.weight_array

    def get_max_index(self, result):
        max, index = result[0],0
        for idx in range(len(result)):
            if result[idx] > max:
                max, index = result[idx], idx
        return index

    def Y_list(self, one_hot_list):
        y_list = []
        for i in one_hot_list:
            sentence_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            for j in i.word_index:
                sentence_result += np.array(self.weight_array[j])
            y_list.append(sentence_result)
        return y_list

    def y_list(self, list):
        sentence_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        for i in list:
            sentence_result += np.array(self.weight_array[i])
        return sentence_result


    def set_batchBlock(self, examples):
        if len(examples) % self.hyperparameter_1.batch_size == 0:
            batchBlock = len(examples) // self.hyperparameter_1.batch_size
        else:
            batchBlock = len(examples) // self.hyperparameter_1.batch_size + 1
        return batchBlock

    def start_and_end_pos(self, every_batchBlock, train_exam_list):
        start_pos = every_batchBlock * self.hyperparameter_1.batch_size
        end_pos = (every_batchBlock + 1) * self.hyperparameter_1.batch_size
        if end_pos >= len(train_exam_list):
            end_pos = len(train_exam_list)
        return start_pos, end_pos

    def count_loss(self, y):
        p = np.max(y)
        return -1 * np.log(p)

    def softmax(self, result):
        result_list = []
        bottom = 0
        max_idx = self.get_max_index(result)
        for index, value in enumerate(result):
            bottom += np.exp(value - result[max_idx])
        for index, value in enumerate(result):
            result_list.append(np.exp(value - result[max_idx])/bottom)
        return result_list


    def train(self, train_Inst,dev_Inst,test_Inst):
        # a = self.creat_feature_alphabet(self.extract_sentence_feature_and_label_encoding(train_Inst))
        # print(a)
        feat_alphabet = self.creat_feature_alphabet(self.extract_sentence_feature_and_label_encoding(train_Inst,train_tree_path))
        print(len(feat_alphabet.dict))

        train_exam_list = self.one_hot_encoding(train_Inst, feat_alphabet,train_tree_path)
        dev_exam_list = self.one_hot_encoding(dev_Inst, feat_alphabet, dev_tree_path)
        test_exam_list = self.one_hot_encoding(test_Inst, feat_alphabet, test_tree_path)
        # self.save_depend_tree(test_parser_sentence)
        self.last_updata = [0.0 for i in range(len(feat_alphabet.list))]
        self.weight_array = np.zeros((len(feat_alphabet.list), self.hyperparameter_1.class_num))
        self.average_w_array = np.zeros((len(feat_alphabet.list), self.hyperparameter_1.class_num))
        self.out_w_array = np.zeros((len(feat_alphabet.list), self.hyperparameter_1.class_num))
        train_size = len(train_exam_list)
        max_updata = 1
        for epoch in range(1, self.hyperparameter_1.epochs + 1):
            print("————第{}轮迭代，共{}轮————Time = {}".format(epoch, self.hyperparameter_1.epochs, time.time()))
            corrects, accuracy, sum, steps, all_loss, c = 0, 0, 0, 0, 0, 0
            random.shuffle(train_exam_list)
            batchBlock = self.set_batchBlock(train_exam_list)
            for every_batchBlock in range(batchBlock):
                start_pos, end_pos = self.start_and_end_pos(every_batchBlock, train_exam_list)
                sentence_result = self.Y_list(train_exam_list[start_pos:end_pos])
                for idx in range(len(sentence_result)):
                    steps += 1
                    sum += 1
                    word_max_index = self.get_max_index(sentence_result[idx])
                    if word_max_index != train_exam_list[start_pos + idx].max_label_index:
                        all_loss += 1
                        for i in train_exam_list[start_pos + idx].word_index:
                            times = max_updata - self.last_updata[i]
                            self.average_w_array[i] += self.weight_array[i] * times
                            self.weight_array[i][self.get_max_index(sentence_result[idx])] -= 1
                            self.weight_array[i][train_exam_list[start_pos + idx].max_label_index] += 1
                            self.average_w_array[i] += self.weight_array[i]
                        # for i in train_exam_list[start_pos + idx].word_index:
                        #     times = max_updata - self.last_updata[i]
                        #     self.weight_array[i] += np.array(train_exam_list[start_pos + idx].label_index)
                        #     self.average_w_array[i] += np.array(train_exam_list[start_pos + idx].label_index) * max_updata
                        for i in train_exam_list[start_pos + idx].word_index:
                            self.last_updata[i] = max_updata
                    else:
                        # exam.append(train_exam_list[start_pos + idx].word_index)
                        corrects += 1
                    max_updata += 1
                    # for idx in range(len(exam)):
                    #     for i in exam[idx]:
                    #         times = max_updata - self.last_updata[i]
                    #         self.average_w_array[i] += np.array(self.weight_array[i]) * times
                    # for i in range(len(train_exam_list)):
                    #     self.weight_array[i] -= self.average_w_array[i] / max_updata
            # self.out_w_array= self.weight_array - (np.array(self.average_w_array) / max_updata)

            for i in range(len(train_exam_list)):
                times = max_updata - self.last_updata[i]
                self.average_w_array[i] += self.weight_array[i] * times

            if steps % self.hyperparameter_1.log_interval == 0:
                accuracy = corrects / sum * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                train_size,
                                                                                all_loss,
                                                                                accuracy,
                                                                                corrects,
                                                                                sum))
            if steps % self.hyperparameter_1.test_interval == 0:
                self.eval(dev_exam_list)

    def eval(self, dev_exam_list):
        corrects, accuracy, sum = 0, 0, 0
        train_size = len(dev_exam_list)
        for idx in range(len(dev_exam_list)):
            sum += 1
            sentence_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            for i in dev_exam_list[idx].word_index:
                sentence_result += np.array(self.average_w_array[i])
            word_max_index = self.get_max_index(sentence_result)
            if word_max_index == dev_exam_list[idx].max_label_index:
                corrects += 1
        accuracy = corrects / sum * 100.0
        print('\nEvaluation -  acc: {:.4f}%({}/{}) \n'.format(accuracy,
                                                              corrects,
                                                              train_size))


a = Classifier()
train_Inst, train_parser_sentence = a.read_file(path='data/raw.clean.train')
#train_Inst = a.read_file(path='data/raw.clean.test')
# train_Inst, parser_sentence = a.read_file(path='data/train_data')
dev_Inst, dev_parser_sentence = a.read_file(path='data/raw.clean.dev')
test_Inst, test_parser_sentence = a.read_file(path='data/raw.clean.test')

train_tree_path = "Factored-model/train_depend_tree.txt"
dev_tree_path = "Factored-model/dev_depend_tree.txt"
test_tree_path = "Factored-model/test_depend_tree.txt"
a.train(train_Inst, dev_Inst, test_Inst)
