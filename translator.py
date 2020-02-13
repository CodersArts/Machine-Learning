import keras
import pydotplus
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
import string
import re
from pickle import dump,load
from unicodedata import normalize
from numpy import array
from numpy.random import rand
from numpy.random import 
import sys

loc = sys.argv[0]
destination = sys.argv[1]
destination_full  = sys.argv[2]
destination_train = sys.argv[3]
destination_test  = sys.argv[4]

file = open(loc, mode='rt', encoding='utf-8')
text = file.read()
file.close()

lines = text.strip().split('\n')
en_ge = [line.split('\t') for line in  lines]
cleaned = list()
re_print = re.compile('[^%s]' % re.escape(string.printable))
table = str.maketrans('', '', string.punctuation)

for instance in en_ge:
        clean = list()
        for row in instance:
            row = normalize('NFD', row).encode('ascii', 'ignore')
            row = row.decode('UTF-8')
            row = row.split()
            row = [word.lower() for word in row]
            row = [word.translate(table) for word in row]
            row = [re_print.sub('', w) for w in row]
            row = [word for word in row if word.isalpha()]
            clean.append(' '.join(row))
        cleaned.append(clean)
cleaned = array(cleaned)

raw_data = load(open(destination,'rb'))
max_count = 10000

dataset = raw_data[:max_count, :]

shuffle(dataset)

train, test = dataset[:9000], dataset[9000:]

dump(dataset, open(destination_full , 'wb'))
dump(train, open(destination_train, 'wb'))
dump(test, open(destination_test , 'wb'))

dataset = load(open(destination_full, 'rb'))
train = load(open(destination_train, 'rb'))
test = load(open(destination_test, 'rb'))

def max_length(lines):
    return max(len(line.split()) for line in lines)keras.utils.vis_utils.pydot = pydot

plot_model(model, to_file='/home/codersarts/Desktop/data/NTMmodel.png')

tokenizerEN = Tokenizer()
tokenizerEN.fit_on_texts(dataset[:,0])
eng_vocab_size = len(tokenizerEN.word_index)+1
eng_length = max_length(dataset[:, 0])

print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

tokenizerGE = Tokenizer()
tokenizerGE.fit_on_texts(dataset[:,1])
ger_vocab_size = len(tokenizerGE.word_index)+1
ger_length = max_length(dataset[:, 1])

print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

x = tokenizerGE.texts_to_sequences(train[:,1])
TrainX = pad_sequences(x , maxlen = ger_length , padding = 'post')

y = tokenizerEN.texts_to_sequences(train[:,0])
trainY = pad_sequences(y , maxlen = eng_length , padding = 'post')

yl = list()

for seq in trainY:
    encod = to_categorical(seq , num_classes = eng_vocab_size)
    yl.append(encod)

var = array(yl)

TrainY = var.reshape(trainY.shape[0], trainY.shape[1], eng_vocab_size) 
x = tokenizerGE.texts_to_sequences(test[:,1])
TestX = pad_sequences(x , maxlen = ger_length , padding = 'post')

y = tokenizerEN.texts_to_sequences(test[:,0])
testY = pad_sequences(y , maxlen = eng_length , padding = 'post')

yl = list()

for seq in testY:
    encod = to_categorical(seq , num_classes = eng_vocab_size)
    yl.append(encod)

var = array(yl)
TestY = var.reshape(testY.shape[0], testY.shape[1], eng_vocab_size)

model = Sequential()
model.add(Embedding(ger_vocab_size, 256, input_length = ger_length, mask_zero=True))
model.add(LSTM(256))
model.add(RepeatVector(eng_length))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(eng_vocab_size, activation='softmax')))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(TrainX, TrainY, epochs=29, batch_size=64, validation_data=(TestX, TestY))

keras.utils.vis_utils.pydot = pydot
plot_model(model, to_file=sys.argv[5])