from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
import pandas as pd
bert_dev_pre = pd.read_csv('root/BERT_310k_MolWt_pre.csv')['dev_pre']

bert_test_pre = pd.read_csv('root/BERT_310k_MolWt_pre.csv')['test_pre']

roberta_dev_pre = pd.read_csv('root/RoBERTa_310k_MolWt_pre.csv')['dev_pre']

roberta_test_pre = pd.read_csv('root/RoBERTa_310k_MolWt_pre.csv')['test_pre']

xlnet_dev_pre = pd.read_csv('root/XLNet_310k_MolWt_pre.csv')['dev_pre']

xlnet_test_pre = pd.read_csv('root/XLNet_310k_MolWt_pre.csv')['test_pre']
# label
dev_label = pd.read_csv('root/XLNet_310k_MolWt_pre.csv')['dev_label']
test_label = pd.read_csv('root/XLNet_310k_MolWt_pre.csv')['test_label'].tolist()
# Feature
meta_features = np.column_stack((bert_dev_pre,roberta_dev_pre,xlnet_dev_pre))
val_true_labels = dev_label

test_meta_features = np.column_stack((bert_test_pre,roberta_test_pre,xlnet_test_pre))
test_true_labels = test_label
# 假设val_predictions是验证集的预测值，val_true_labels是验证集的真实标签
# test_predictions是测试集的预测值，test_true_labels是测试集的真实标签

# 创建BaggingRegressor作为元学习器
print('310k_MolWt')

bagging_meta_model = BaggingRegressor(
                n_estimators=100,
                max_samples=0.5,
                max_features=1.0,
                random_state=42)

# 在验证集的元特征上训练元学习器
bagging_meta_model.fit(meta_features, val_true_labels)

# 使用测试集的预测值作为元特征进行预测
meta_predictions = bagging_meta_model.predict(test_meta_features)

# 计算MAE和RMSE
mae = mean_absolute_error(test_true_labels, meta_predictions)

rmse = np.sqrt(mean_squared_error(test_true_labels, meta_predictions))
r2 = r2_score(meta_predictions,test_true_labels)
print(f"元学习器的MAE: {mae}")
print(f"元学习器的RMSE: {rmse}")
print(f'r2:{r2}')
