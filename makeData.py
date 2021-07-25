import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from bert.config import MAX_SEQ_LENGTH

class DataMaker :
    def __init__(self):
        self.x_data = list()
        self.y_data = list()
        self.x_train = list()
        self.x_test = list()
        self.y_train = list()
        self.y_test = list()

    def import_data(self):

        benign_path = "C:/Users/ahoho/Downloads/KISA-CISC2017-Malware-2nd/0/*"
        malware_path = "C:/Users/ahoho/Downloads/KISA-CISC2017-Malware-2nd/1/*"

        benign_file_list = glob.glob(benign_path)
        malware_file_list = glob.glob(malware_path)

        for file in benign_file_list:
            with open(file) as f :
                apis = f.read().splitlines()
                self.x_data.append(apis)
                self.y_data.append(0)

        for file in malware_file_list:
            with open(file) as f :
                apis = f.read().splitlines()
                self.x_data.append(apis)
                self.y_data.append(1)

    def split_train_test(self):
        self.x_train ,self.x_test, self.y_train, self.y_test \
            = train_test_split(self.x_data,self.y_data, test_size=0.25, shuffle=True, random_state=1234)

    def preprocess(self):
        for i in range(len(self.x_train)) :
            self.x_train[i] = ['CLS'] + self.x_train[i] + ['SEP']

        for i in range(len(self.x_test)) :
            self.x_test[i] = ['CLS'] + self.x_test[i] + ['SEP']

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)

        with open('./data/dict_api_index.pickle','wb') as outfile:
            pickle.dump(tokenizer.word_index, outfile)

        for i in range(len(self.x_train)):
            if(len(self.x_train[i])>MAX_SEQ_LENGTH):
                self.x_train[i] = self.x_train[i][:MAX_SEQ_LENGTH-1] + ['SEP']
                print(self.x_train[i])

        for i in range(len(self.x_test)):
            if (len(self.x_test[i]) > MAX_SEQ_LENGTH):
                self.x_test[i] = self.x_test[i][:MAX_SEQ_LENGTH-1] + ['SEP']

        self.x_train = tokenizer.texts_to_sequences(self.x_train)
        self.x_test = tokenizer.texts_to_sequences(self.x_test)

    def save_data(self, file_path):
        train_data = list()
        test_data = list()
        for i in range(len(self.x_train)):
            train_data.append([self.x_train[i], self.y_train[i]])
        for i in range(len(self.x_test)):
            test_data.append([self.x_test[i], self.y_test[i]])

        train_data = np.array(train_data)
        test_data = np.array(test_data)

        np.save(file_path+"/train_data.npy", train_data)
        np.save(file_path+"/test_data.npy", test_data)


data = DataMaker()
data.import_data()
data.split_train_test()
data.preprocess()
data.save_data('./data')
