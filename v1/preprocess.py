import numpy as np
import pandas as pd
import csv

new_tetfp = './data/TBrain_Round2_DataSet_20180601/tetfp.csv'


def getSeq(timeStep):
    df = pd.read_csv(new_tetfp, sep=',',
                     dtype={'代碼': np.int32, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')
    df2 = pd.read_csv('./data/three_big2.csv', encoding='utf-8-sig')
    for i in range(3):
        mean = np.mean(df2[[df2.columns[4+3*i], df2.columns[5+3*i]]].values.reshape([-1]))
        std = np.std(df2[[df2.columns[4+3*i], df2.columns[5+3*i]]].values.reshape([-1]))
        df2[df2.columns[4+3*i]] = (df2[df2.columns[4+3*i]]-mean)/std
        df2[df2.columns[5+3*i]] = (df2[df2.columns[5+3*i]]-mean)/std

    df['中文簡稱'] = df['中文簡稱'].str.split().apply(lambda x: x[0].replace(' ', ''))
    stock_names = df['中文簡稱'].unique()
    mean_deal = df['成交張數(張)'].mean()
    std_deal = df['成交張數(張)'].std()
    df['成交張數(張)'] = (df['成交張數(張)']-mean_deal)/std_deal
    seq = []
    seq_y = []
    for num, name in enumerate(stock_names):
        stock_mask = df['中文簡稱']==name
        stock_mask2 = df2['證券名稱']==name
        data_mask = df[stock_mask]
        data_mask2 = df2[stock_mask2]
        print('Preprocess {} data!'.format(name))
        for i in range(data_mask.shape[0]-(timeStep+2)):
            tmp = []
            for j in range(timeStep):
                data1 = data_mask.iloc[i+j]
                data2 = data_mask.iloc[i+j+1]
                data3 = data_mask2.iloc[i+j]
                data4 = data_mask2.iloc[i+j+1]
                tmp.append([data2['開盤價(元)']-data1['開盤價(元)'], data2['最高價(元)']-data1['最高價(元)'], data2['最低價(元)']-data1['最低價(元)'], data2['收盤價(元)']-data1['收盤價(元)'], data2['成交張數(張)']-data1['成交張數(張)'],
                            data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'], data4['自營商買進股數'], data4['自營商賣出股數']])
            data1 = data_mask.iloc[i+timeStep]
            data2 = data_mask.iloc[i+timeStep+1]
            data4 = data_mask2.iloc[i+timeStep+1]
            seq_y.append([data2['開盤價(元)']-data1['開盤價(元)'], data2['最高價(元)']-data1['最高價(元)'], data2['最低價(元)']-data1['最低價(元)'], data2['收盤價(元)']-data1['收盤價(元)'], data2['成交張數(張)']-data1['成交張數(張)'],
                          data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'], data4['自營商買進股數'], data4['自營商賣出股數']])
            seq.append(tmp)
    return seq, seq_y
def getSeq2(timeStep, lower, upper):
    df = pd.read_csv(new_tetfp, sep=',',
                     dtype={'代碼': np.int32, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')
    df2 = pd.read_csv('./data/three_big2.csv', encoding='utf-8-sig')
    # normalize data
    for i in range(3):
        mean = np.mean(df2[[df2.columns[4 + 3 * i], df2.columns[5 + 3 * i]]].values.reshape([-1]))
        std = np.std(df2[[df2.columns[4 + 3 * i], df2.columns[5 + 3 * i]]].values.reshape([-1]))
        df2[df2.columns[4 + 3 * i]] = (df2[df2.columns[4 + 3 * i]] - mean) / std
        df2[df2.columns[5 + 3 * i]] = (df2[df2.columns[5 + 3 * i]] - mean) / std

    df['中文簡稱'] = df['中文簡稱'].str.split().apply(lambda x: x[0].replace(' ', ''))
    stock_names = df['中文簡稱'].unique()
    mean_deal = df['成交張數(張)'].mean()
    std_deal = df['成交張數(張)'].std()
    df['成交張數(張)'] = (df['成交張數(張)'] - mean_deal) / std_deal
    df2['日期'] = df['日期'].astype(str)
    # normalize data end

    df = df[(df['日期']>lower) & (df['日期']<upper)]
    df2 = df2[ (df2['日期'] > lower) & (df2['日期'] < upper)]
    seq = []
    seq_y = []
    for num, name in enumerate(stock_names):
        stock_mask = df['中文簡稱'] == name
        stock_mask2 = df2['證券名稱'] == name
        data_mask = df[stock_mask]
        data_mask2 = df2[stock_mask2]
        print('Preprocess {} data!'.format(name))
        for i in range(data_mask.shape[0] - (timeStep + 2)):
            tmp = []
            for j in range(timeStep):
                data1 = data_mask.iloc[i + j]
                data2 = data_mask.iloc[i + j + 1]
                data3 = data_mask2.iloc[i + j]
                data4 = data_mask2.iloc[i + j + 1]
                tmp.append([data2['開盤價(元)'] - data1['開盤價(元)'], data2['最高價(元)'] - data1['最高價(元)'],
                            data2['最低價(元)'] - data1['最低價(元)'], data2['收盤價(元)'] - data1['收盤價(元)'],
                            data2['成交張數(張)'] - data1['成交張數(張)'],
                            data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'], data4['自營商買進股數'],
                            data4['自營商賣出股數']])
            data1 = data_mask.iloc[i + timeStep]
            data2 = data_mask.iloc[i + timeStep + 1]
            data4 = data_mask2.iloc[i + timeStep + 1]
            seq_y.append([data2['開盤價(元)'] - data1['開盤價(元)'], data2['最高價(元)'] - data1['最高價(元)'],
                          data2['最低價(元)'] - data1['最低價(元)'], data2['收盤價(元)'] - data1['收盤價(元)'],
                          data2['成交張數(張)'] - data1['成交張數(張)'],
                          data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'], data4['自營商買進股數'],
                          data4['自營商賣出股數']])
            seq.append(tmp)
    return seq, seq_y

def getStock(stock, timeStep):
    df = pd.read_csv(new_tetfp, sep=',',
                     dtype={'代碼': np.str, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')
    df2 = pd.read_csv('./data/three_big2.csv', encoding='utf-8-sig')
    map_dict = {'元大台灣50': 50, '元大中型100': 51, '富邦科技': 52, '元大電子': 53, '元大台商50': 54, '元大MSCI金融': 55,
                '元大高股息': 56, '富邦摩台': 57, '富邦發達': 58, '富邦金融': 59, '元大富櫃50': 6201, '元大MSCI台灣': 6203,
                '永豐臺灣加權': 6204, '富邦台50': 6208, '兆豐藍籌30': 690, '富邦公司治理': 692, '國泰臺灣低波動30': 701,
                '元大台灣高息低波': 713}
    df2['證券名稱'] = df2['證券名稱'].map(map_dict).astype(str)
    df2['日期'] = df2['日期'].astype(str)
    for i in range(3):
        mean = np.mean(df2[[df2.columns[4 + 3 * i], df2.columns[5 + 3 * i]]].values.reshape([-1]))
        std = np.std(df2[[df2.columns[4 + 3 * i], df2.columns[5 + 3 * i]]].values.reshape([-1]))
        df2[df2.columns[4 + 3 * i]] = (df2[df2.columns[4 + 3 * i]] - mean) / std
        df2[df2.columns[5 + 3 * i]] = (df2[df2.columns[5 + 3 * i]] - mean) / std
    mean_deal = df['成交張數(張)'].mean()
    std_deal = df['成交張數(張)'].std()
    data = df[df['代碼']==stock]
    data = data[-70:]
    data_mask = df2[df2['證券名稱']==stock]
    seq = []
    seq_y = []
    for i in range(data.shape[0]-(timeStep+2)):
        tmp = []
        for j in range(timeStep):
            data1 = data.iloc[i+j]
            data2 = data.iloc[i+j+1]
            data4 = data_mask.iloc[i+j+1]
            open = data2['開盤價(元)'] - data1['開盤價(元)']
            high = data2['最高價(元)'] - data1['最高價(元)']
            low = data2['最低價(元)'] - data1['最低價(元)']
            close = data2['收盤價(元)'] - data1['收盤價(元)']
            vol = ((data2['成交張數(張)'] - data1['成交張數(張)']) - mean_deal) / std_deal
            tmp.append([open, high, low, close, vol, data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'],
                        data4['自營商買進股數'], data4['自營商賣出股數']])
        data1 = data.iloc[i+timeStep]
        data2 = data.iloc[i+timeStep+1]
        seq_y.append([data2['開盤價(元)']-data1['開盤價(元)'], data2['最高價(元)']-data1['最高價(元)'], data2['最低價(元)']-data1['最低價(元)'], data2['收盤價(元)']-data1['收盤價(元)'], data2['成交張數(張)']-data1['成交張數(張)'],
                      data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'], data4['自營商買進股數'], data4['自營商賣出股數']])
        seq.append(tmp)
    return seq, seq_y

def getBatch(seq, seq_y, sample_index, base, batch_size):
    tr = []
    gt = []
    for i in range(batch_size):
        tr.append(seq[sample_index[base*batch_size+i]])
        gt.append(seq_y[sample_index[base*batch_size+i]])
    return tr, gt
def transform_file():
    # 轉換big_three內容格式，以符合我的code讀取
    df = pd.read_csv('./data/three_big.csv')
    df['三大法人買賣超股數'] = df['三大法人買賣超股數'].str.split().apply(lambda x: x[0].replace(',', ''))
    df['三大法人買賣超股數'] = df['三大法人買賣超股數'].astype(np.float64)
    df['投信買賣超股數'] = df['投信買賣超股數'].str.split().apply(lambda x: x[0].replace(',', ''))
    df['投信買賣超股數'] = df['投信買賣超股數'].astype(np.float64)
    df['投信買進股數'] = df['投信買進股數'].str.split().apply(lambda x: x[0].replace(',', ''))
    df['投信買進股數'] = df['投信買進股數'].astype(np.float64)
    df['投信賣出股數'] = df['投信賣出股數'].str.split().apply(lambda x: x[0].replace(',', ''))
    df['投信賣出股數'] = df['投信賣出股數'].astype(np.float64)
    df.to_csv('./three_big2.csv', encoding='utf-8-sig', index=False)


if __name__ == '__main__':
    transform_file()
    map_dict = {'元大台灣50': 50, '元大中型100': 51, '富邦科技': 52, '元大電子': 53, '元大台商50': 54, '元大MSCI金融': 55,
                '元大高股息': 56, '富邦摩台': 57, '富邦發達': 58, '富邦金融': 59, '元大富櫃50': 6201, '元大MSCI台灣': 6203,
                '永豐臺灣加權': 6204, '富邦台50': 6208, '兆豐藍籌30': 690, '富邦公司治理': 692, '國泰臺灣低波動30': 701,
                '元大台灣高息低波': 713}
    #a,b = getSeq2( 6, '20130101', '20131231')
    df = pd.read_csv(new_tetfp, sep=',',
                     dtype={'代碼': str, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')