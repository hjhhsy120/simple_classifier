import scipy.io as sio
import pickle
for data_name in ['tpcc_16w', 'tpcc_500w', 'tpce_3000']:
    data = sio.loadmat('dbsherlock_dataset_' + data_name + '.mat')
    fea = []
    for i in range(10):
        for j in range(11):
            fea.append(data['test_datasets'][i][j][0][0][0])
    lab = []
    for i in range(10):
        for j in range(11):
            lab.append(i)
    abn = []
    for i in range(10):
        for j in range(11):
            abn.append([data['abnormal_regions'][i][j][0][0], data['abnormal_regions'][i][j][0][-1]+1])
    with open(data_name + '.pkl' , 'wb') as f:
        pickle.dump((fea, lab, abn),f)