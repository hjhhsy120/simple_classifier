代码说明：
将dbsherlock提供的数据（单异常类型）视为多分类问题，将正常时间段、异常时间段分别计算各项指标的平均值，两者拼接作为输入特征，异常类别作为标签，按7:2:2划分训练、验证、测试集（每种类别数量均匀），训练lightgbm模型，打印测试集准确率和全体数据上的准确率，可以达到或接近100%
执行方式：
从https://github.com/dongyoungy/dbsherlock-reproducibility下载数据集。
将3个.mat文件放在本目录下，先执行gendata.py进行数据预处理，再执行learn.py进行训练和测试。需learn.py的data_name变量为数据集名，即'tpcc_16w', 'tpcc_500w', 'tpce_3000'之一。