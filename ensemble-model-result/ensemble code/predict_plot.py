import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
data = pd.read_csv('/Users/macbook/Desktop/code/ensemble-model-result/ensemble_250k_logP_result.csv')
preds_test = data['final_pre'].values.tolist()
true_test = data['label'].values.tolist()


r_squared = r2_score(true_test, preds_test)
rmse = mean_squared_error(true_test, preds_test) ** 0.5
mae = mean_absolute_error(true_test, preds_test)

def make_plot(true, pred, rmse, r2_score, mae, name):
    fontsize = 12
    fig, ax = plt.subplots(figsize=(8, 8))
    r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2_score), color="#1E90FF")  # 深蓝色
    rmse_patch = mpatches.Patch(label="RMSE = {:.4f}".format(rmse), color="#00BFFF")  # 中蓝色
    mae_patch = mpatches.Patch(label="MAE = {:.4f}".format(mae), color="#87CEEB")  # 浅蓝色

    plt.xlim(-7, 9)
    plt.ylim(-7, 9)
    plt.scatter(true, pred, alpha=0.2, color="#1E90FF")
    plt.plot(np.arange(-7, 9, 0.01), np.arange(-7, 9, 0.01), ls="--", c=".3")
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=fontsize)
    ax.set_xlabel('True Label:LogP', fontsize=fontsize)
    ax.set_ylabel('Predicted:LogP', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    return fig

print(f"  R2 {r_squared:.3f},RMSE {rmse:.2f},MAE {mae:.2f}")

fig = make_plot(true_test, preds_test, rmse, r_squared, mae, ' ')
fig.savefig('/Users/macbook/Desktop/论文图/250k_logP.tiff', dpi=300)

def make_hist(trues, preds):
    fontsize = 12
    fig_hist, ax = plt.subplots(figsize=(8, 8))

    plt.hist(trues, bins=30, label='true',
                     facecolor='#1E90FF', histtype='bar', alpha=0.8)
    plt.hist(preds, bins=30, label='predict',
                     facecolor='#FFA500', histtype='bar', alpha=0.6)

    plt.xlabel('LogP', fontsize=fontsize)
    plt.ylabel('amount', fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=fontsize)
    # ax.set_title(fontsize=fontsize)
    return fig_hist

fig_hist = make_hist(true_test, preds_test)
fig_hist.savefig('/Users/macbook/Desktop/论文图/250k_logP_hist.tiff', dpi=300)


# plt.show()