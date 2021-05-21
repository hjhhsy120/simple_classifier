
import pickle
import lightgbm as lgb
import numpy as np
import random
data_name = 'tpcc_16w'
with open(data_name + '.pkl' , 'rb') as f:
    data = pickle.load(f)
fea, lab, abn = data
fea2 = []
for i in range(110):
    fe1 = np.mean(np.append(fea[i][ : abn[i][0]], fea[i][abn[i][1] : ]), axis=0)
    fe2 = np.mean(fea[i][abn[i][0] : abn[i][1]], axis=0)
    fea2.append(list(np.append(fe1, fe2)))
testidx = set()
valididx = set()
for i in range(10):
    tt = random.sample([11*i + x for x in range(11)], 4)
    testidx.add(tt[0])
    testidx.add(tt[1])
    valididx.add(tt[2])
    valididx.add(tt[3])
trainfea = []
trainlab = []
validfea = []
validlab = []
testfea = []
testlab = []
for i in range(110):
    if i in testidx:
        testfea.append(fea2[i])
        testlab.append(lab[i])
    elif i in valididx:
        validfea.append(fea2[i])
        validlab.append(lab[i])
    else:
        trainfea.append(fea2[i])
        trainlab.append(lab[i])

train_data = lgb.Dataset(trainfea, label=trainlab)
valid_data = lgb.Dataset(validfea, label=validlab)

params = {
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':8,
    'objective':'multiclass',
    'num_class':10,
}
clf = lgb.train(params, train_data, valid_sets=[valid_data])
result = clf.predict(testfea)
testpred = [list(x).index(max(x)) for x in result]
good = 0
for i in range(len(testlab)):
    if testpred[i] == testlab[i]:
        good += 1
print('test:', good/len(testlab))

result = clf.predict(fea2)
pred = [list(x).index(max(x)) for x in result]
good = 0
for i in range(len(lab)):
    if pred[i] == lab[i]:
        good += 1
print('test:', good/len(lab))