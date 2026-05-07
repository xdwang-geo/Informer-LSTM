import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 数据读取，预处理
def data_preprocessing(dir, filename, target):
    '''
    参数
    :param dir: 文件路径
    :param filename: 数据文件名称 xxx.csv   第一列 一定是 时间！！！
    :param target: 预测的目标变量名称
    :return:
    df_normalized：归一化后的原始数据框
    target_values：归一化后的目标变量
    '''
    file_path =  os.path.join(dir, filename)
    original_data = pd.read_csv(file_path)

    # 分离时间列和其他列
    time_col = original_data.iloc[:, 0]  # 获取时间列
    data_cols = original_data.iloc[:, 2:]  # 获取其余列
    # 修改时间列的列名
    time_col = time_col.rename('date')

    # 拿出目标列对其进行处理
    target_values_col = original_data[target]
    target_values = np.array(target_values_col.tolist())  # 转换为numpy
    target_values = target_values.reshape(-1, 1)
    # 归一化处理
    # 使用标准化（z-score标准化）
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_cols)
    target_values = scaler.fit_transform(target_values)
    # 保存 归一化 模型
    dump(scaler, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\scaler')

    # 将归一化后的数据合并回原始数据框
    normalized_df = pd.DataFrame(normalized_data, columns=data_cols.columns)
    df_normalized = pd.concat([time_col, normalized_df], axis=1)
    target_values = target_values.reshape(-1)
    df_normalized[target] = target_values

    return df_normalized



# 通过滑动窗口制作多步预测数据集
def create_multistep_dataset(data, window_size, label_len, forecast_step, task_type='MS'):
    '''
    参数：
    :param data: 数据元组（特征数据，标签数据）， 单变量--（特征数据，）
    :param window_size:  样本窗口大小
    :param label_len:    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    :param forecast_step: 多步预测 步数
    :param task_type:  任务类型(数据格式：字符串类型)  S：单变量预测单变量，MS：多变量预测单变量，默认 'MS'
    :return:
        sample_features  : 特征数据
        labels           : 签数据
    '''
    sample_features = []
    labels = []

    # 第一种任务 MS：多变量预测单变量
    if task_type == 'MS':
        features = data.values
        ylabel = data.values
        for i in range(len(ylabel) - window_size - forecast_step + 1):
            sample_features.append(features[i:i + window_size, :])
            labels.append(ylabel[i + window_size - label_len:i + window_size + forecast_step, :])

    # 第二种任务 S：单变量预测单变量
    elif task_type == 'S':
        features = data.values  # (24528, 1)
        for i in range(len(features) - window_size - forecast_step + 1):
            sample_features.append(features[i:i + window_size, :])
            labels.append(features[i + window_size- label_len:i + window_size + forecast_step, :])

    # 将列表转换为单一的NumPy数组
    sample_features = np.array(sample_features)
    labels = np.array(labels)

    return sample_features, labels


# 制作多步预测数据集
def make_dataset(df_normalized, target, window_size, label_len, forecast_step, task_type='MS', split_rate=[0.8, 0.1, 0.1]):
    '''
    参数
    :param df_normalized:  归一化后的 CSV 数据！
    :param window_size:    数据滑动窗口值
    :param label_len:    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    :param forecast_step:  预测步数
    :param task_type: 任务类型(数据格式：字符串类型)  S：单变量预测单变量，MS：多变量预测单变量，默认 'MS'
    :param split_rate:     数据划分比例
    :return:
           train_xdata: 训练集数据
           train_ylabel: 训练集标签
           valid_xdata: 验证集数据
           valid_ylabel: 验证集标签
           test_xdata: 测试集数据
           test_ylabel: 测试集标签
    '''
    # 第一步，划分数据集
    sample_len = df_normalized.shape[0]  # 样本总长度
    train_len = int(sample_len * split_rate[0])  # 向下取整
    valid_len = int(sample_len * split_rate[1])  # 验证集长度
    test_len = sample_len - train_len - valid_len

    # 第一种任务 MS：多变量预测单变量
    if task_type == 'MS':
        train_data = df_normalized.iloc[:train_len, :]  # 训练集
        valid_data = df_normalized.iloc[train_len:train_len + valid_len, :]  # 验证集
        test_data = df_normalized.iloc[train_len + valid_len:, :]   # 测试集
        test_start_date = test_data.iloc[0, 0]  # 假设第一列是日期列
        print(f"测试集开始日期: {test_start_date}")

        #在制作特征数据集时，去除目标列
        train_features = train_data.drop(columns=[target])
        valid_features = valid_data.drop(columns=[target])
        test_features = test_data.drop(columns=[target])

        # 第二步，制作数据集标签  滑动窗口
        train_xdata, train_ylabel = create_multistep_dataset(train_data, window_size, label_len, forecast_step, task_type)
        valid_xdata, valid_ylabel = create_multistep_dataset(valid_data, window_size, label_len, forecast_step, task_type)
        test_xdata, test_ylabel = create_multistep_dataset(test_data, window_size, label_len, forecast_step, task_type)

    # 第二种任务 S：单变量预测单变量
    elif task_type == 'S':
        target_values = df_normalized[['date', target]]
        train_data = target_values.iloc[:train_len, :]  # 训练集 标签
        valid_data = target_values.iloc[train_len:, :]  # 训练集 标签
        test_data = target_values.iloc[train_len:, :]
        # 第二步，制作数据集标签  滑动窗口
        train_xdata, train_ylabel = create_multistep_dataset(train_data, window_size, label_len, forecast_step, task_type)
        valid_xdata, valid_ylabel = create_multistep_dataset(valid_data, window_size, label_len, forecast_step, task_type)
        test_xdata, valid_ylabel = create_multistep_dataset(test_data, window_size, label_len, forecast_step, task_type)

    # 参数错误
    else:
        print("task_type ERROR!")
        return

    return train_xdata, train_ylabel, valid_xdata,valid_ylabel, test_xdata,test_ylabel

# 自定义数据集
class MyData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    def create_time(self, data):
        # 提取时间列
        time = data[:, 0]
        time = pd.to_datetime(time)

        # 提取各时间特征
        # week = np.int32(time.dayofweek)[:, None]
        month = np.int32(time.month)[:, None]
        day = np.int32(time.day)[:, None]
        # hour = np.int32(time.hour)[:, None]
        # minute = np.int32(time.minute)[:, None]
        time_data = np.concatenate([month, day], axis=-1)

        return time_data

    def __getitem__(self, item):
        value = self.values[item]
        label = self.labels[item]

        value_t = self.create_time(value)
        label_t = self.create_time(label)

        value = value[:, 2:]
        label = label[:, 2:]

        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t


if __name__ == "__main__":

    # 数据集 文件路径
    dir =r'E:\Snowline_simulation_data\Muzat_Infor_LSTM'
    # 数据文件名称 xxx.csv
    filename = 'simulation data.csv'
    # 预测的目标变量名称
    target = 'Altitude'
    # 数据读取，预处理
    df_normalized = data_preprocessing(dir, filename, target)
    print(df_normalized.shape)  # (6000, 15)
    # 定义序列长度和预测步数
    # 定义窗口大小  ： 用过去 30个步长 ，预测未来 1个 步长  (单步预测)
    window_size = 35
    # Informer 解码器的起始 token 长度, decoder中 输入的没有掩码部分序列长度
    label_len = 35
    # 预测步数
    forecast_step = 1
    # 数据集划分比例
    split_rate = [0.8, 0.1, 0.1]

    # 任务类型 S：单变量预测单变量，MS：多变量预测单变量，默认 'MS'
    task_type = 'MS'

    # 制作数据集
    train_xdata, train_ylabel, valid_xdata, valid_ylabel ,test_xdata, test_ylabel = make_dataset(df_normalized, target, window_size, label_len,
                                                                      forecast_step, task_type)

    # print('数据 形状：')
    print(train_xdata.shape, train_ylabel.shape)
    print(valid_xdata.shape, valid_ylabel.shape)
    print(test_xdata.shape, test_ylabel.shape)
    # 保存数据集
    # 保存数据
    dump(train_xdata, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\train_xdata')
    dump(valid_xdata, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\valid_xdata')
    dump(test_xdata, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\test_xdata')
    dump(train_ylabel, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\train_ylabel')
    dump(valid_ylabel, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\valid_ylabel')
    dump(test_ylabel, r'E:\Snowline_simulation_data\Muzat_Infor_LSTM\test_ylabel')

