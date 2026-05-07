from joblib import dump, load
import numpy as np
import torch
import pandas as pd
import os

from numpy.core.defchararray import upper
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
from matplotlib import MatplotlibDeprecationWarning
from scipy import stats
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入模型定义



# 模型 测试集 测试
def model_test(model, test_loader, pre_len):
    model = model.to(device)
    # 预测数据
    original_data = []
    pre_data = []
    # 每一个epoch结束后，在测试集上验证实验结果。
    with torch.no_grad():
        # 将模型设置为评估模式
        model.eval()
        for x, y, xt, yt in test_loader:
            # 创建一个掩码
            mask = torch.zeros_like(y)[:, -pre_len:, :].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            # print(y.size())  # torch.Size([64, 12, 1])
            # 覆盖掉未来信息
            dec_y = torch.cat([y[:, :-pre_len, :], mask], dim=1)
            # 前向传播
            test_pred = model(x, xt, dec_y, yt)  # torch.Size([64, 3, 1])
            # print(y_pred.size())
            # 损失计算
            # print(y[:, -pre_len:].size()) # torch.Size([64, 3, 1])
            # 使用 squeeze 移除尺寸为 1 的最后一个维度
            test_pred = test_pred.squeeze(-1)
            label = y[:, -pre_len:, -1]

            origin_lable = label.tolist()
            original_data += origin_lable

            test_pred = test_pred.tolist()
            pre_data += test_pred


    # 模型分数
    original_data = np.array(original_data)
    pre_data = np.array(pre_data)
    score = r2_score(original_data, pre_data)
    print('*' * 50)
    print('Informer-LSTM 模型分数--R^2:', score)

    print('*' * 50)
    # 测试集上的预测误差

    return original_data, pre_data

if __name__ == '__main__':

    # 参数设置
    pre_len = 1
    input_dim = 8  # 输入的特征维度

    # 加载模型
    model = torch.load(r'E:\Snowline_simulation_data\Manas_Infor_LSTM\best_model_informer_lstm.pt')

    # 加载测试集
    test_loader = load(r'E:\Snowline_simulation_data\Manas_Infor_LSTM\test_loader')

    # 模型预测
    original_data, pre_data = model_test(model, test_loader, pre_len)

    # 反归一化处理
    # 使用相同的均值和标准差对预测结果进行反归一化处理
    # 反标准化
    scaler = load(r'E:\Snowline_simulation_data\Manas_Infor_LSTM\scaler')
    original_data = scaler.inverse_transform(original_data)
    pre_data = scaler.inverse_transform(pre_data)

    test_mse = mean_squared_error(original_data, pre_data)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(original_data, pre_data)
    print('测试数据集上的均方误差--MSE: ', test_mse)
    print('测试数据集上的均方根误差--RMSE: ', test_rmse)
    print('测试数据集上的平均绝对误差--MAE: ', test_mae)

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.linewidth": 1.2
    })


    # 可视化结果
    plt.figure(figsize=(12, 4), dpi=100)
    plt.plot(original_data, label='True value', color=(0/255, 45/255, 148/255))  # 真实值
    plt.plot(pre_data, label='Prediction', color=(190/255, 0/255, 0/255))  # 预测值
    # plt.title('Parallel prediction model based on Informer LSTM', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    # plt.title('Informer-LSTM parallel training process Visualization', fontsize=16)
    plt.savefig(r'E:\Snowline_simulation_data\Manas_Infor_LSTM\预测结果', dpi=100)
    plt.show()  # 显示 lable

    # 绘制核密度散点图
    original_data = np.array(original_data).flatten()
    pre_data = np.array(pre_data).flatten()

    # 将测试集的 original_data 赋值给 x，预测值 pre_data 赋值给 y
    x = original_data
    y = pre_data
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # 使用 numpy 的 polyfit 进行线性回归
    k, b = np.polyfit(x, y, 1)
    regression_line = k * x + b  # 回归线公式

    # 计算常见评估指标
    BIAS = np.mean(x - y)
    MSE = mean_squared_error(x, y)
    RMSE = np.power(MSE,0.5)
    R2 = r2_score(x, y)
    adjR2 = 1 - ((1 - R2) * (len(x) - 1)) / (len(x) - 2)
    MAE = mean_absolute_error(x, y)
    EV = explained_variance_score(x, y)
    NSE = 1 - (RMSE ** 2 / np.var(x))

    n = 1
    t_value = 1.96
    slope, intercept = np.polyfit(x, y, 1)
    std_err = np.std(y - (slope * x + intercept))
    margin_of_error = t_value * (std_err / np.sqrt(n))
    lower_confidence_bound = slope * x + intercept - margin_of_error
    upper_confidence_bound = slope * x + intercept + margin_of_error
    config = {"font.family": 'Times New Roman', "font.size": 16, "mathtext.fontset": 'stixs'}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    # plt.plot(x,lower_confidence_bound,linestyle='--',color='black',dashes=(4,4),label='95% Prediction Band')
    # plt.plot(x, upper_confidence_bound, linestyle='--', color='black', dashes=(4, 4))
    scatter = ax.scatter(x, y, marker='o', c=z, edgecolors=None, s=15, alpha=0.8)
    cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30,label='Frequency')
    cbar.ax.get_yaxis().set_ticks([])
    plt.plot(x,regression_line, 'black',lw=1.5,label='Regression line')
    ax.grid(True,linestyle='--',alpha=0.2)
    plt.plot([3300, 5200], [3300, 5200], 'red', lw=1.5, linestyle='--', label='1:1 line')
    plt.text(5150,3600,'$R^2=%.3f$' % R2,family='Times New Roman', horizontalalignment='right')
    plt.text(5150, 3500, '$RMSE=%.3f$' % RMSE, family='Times New Roman', horizontalalignment='right')
    plt.text(5150, 3400, '$MAE=%.3f$' % MAE, family='Times New Roman', horizontalalignment='right')
    plt.axis([3350,5200,3350,5200])
    ax.legend(loc='upper left', frameon=False)
    plt.savefig(r'E:\Snowline_simulation_data\Manas_Infor_LSTM\预测结果散点图.png', dpi=300)
    plt.show()

    result_df = pd.DataFrame({
        'Index': np.arange(1, len(original_data) + 1),
        'Predicted': pre_data.flatten(),  # 预测值
        'True Value': original_data.flatten()  # 真实值
    })

    # 保存为 CSV 文件
    output_path = r'E:\Snowline_simulation_data\Manas_Infor_LSTM\预测结果.csv'
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
