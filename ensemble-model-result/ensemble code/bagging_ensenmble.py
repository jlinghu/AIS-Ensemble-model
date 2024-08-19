from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
import pandas as pd
bert_dev_pre = pd.read_csv('/root/333.csv')['dev_pre']
bert_dev_pre = pd.read_csv('/root/333.csv')['dev_pre']

bert_test_pre = pd.read_csv('/root/333.csv')['test_pre']

roberta_dev_pre = pd.read_csv('/root/111.csv')['dev_pre']

roberta_test_pre = pd.read_csv('/root/111.csv')['test_pre']

xlnet_dev_pre = pd.read_csv('/root/222.csv')['dev_pre']

xlnet_test_pre = pd.read_csv('/root/222.csv')['test_pre']
# label
dev_label = pd.read_csv('/root/222.csv')['dev_label']
test_label = pd.read_csv('/root/222.csv')['test_label'].tolist()
# Feature
meta_features = np.column_stack((bert_dev_pre,roberta_dev_pre,xlnet_dev_pre))
val_true_labels = dev_label

test_meta_features = np.column_stack((bert_test_pre,roberta_test_pre,xlnet_test_pre))
test_true_labels = test_label
# 假设val_predictions是验证集的预测值，val_true_labels是验证集的真实标签
# test_predictions是测试集的预测值，test_true_labels是测试集的真实标签

best = 0.5
# 创建BaggingRegressor作为元学习器
print('250k_logP')
for k in range(100,201,50):
    for i in range(1,11,1):
        for j in range(1,11,1):
            if i == 10:
                ni = 1.0
            else:
                ni = i*0.1
            if j == 10:
                nj = 1.0
            else:
                nj = j*0.1
            bagging_meta_model = BaggingRegressor(
                            n_estimators=k,
                            max_samples=ni,
                            max_features=nj,
                            random_state=42)

        # 在验证集的元特征上训练元学习器
            bagging_meta_model.fit(meta_features, val_true_labels)

            # 使用测试集的预测值作为元特征进行预测
            meta_predictions = bagging_meta_model.predict(test_meta_features)

            # 计算MAE和RMSE
            mae = mean_absolute_error(test_true_labels, meta_predictions)
            if mae < best:
                best = mae
                rmse = np.sqrt(mean_squared_error(test_true_labels, meta_predictions))
                r2 = r2_score(meta_predictions,test_true_labels)
                print(f'params:{i*0.1,j*0.1,k}')
                print(f"元学习器的MAE: {mae}")
                print(f"元学习器的RMSE: {rmse}")
                print(f'r2:{r2}')
