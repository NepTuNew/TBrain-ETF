import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

import json
import numpy as np
import pandas as pd
import datetime
import time
import csv
import random


new_tetfp = './data/TBrain_Round2_DataSet_20180615/tetfp.csv'
std = 3692.058083292519
mean = 896.5493046115236

def crawl(stockNo, crawlDate, timeStep):
    seq = []
    raw = []
    url = 'http://www.tse.com.tw/exchangeReport/STOCK_DAY?response=json&date=20180401&stockNo=0050&_=1525142113939'
    old_date = '20180401'
    old_num = '0050'
    df = pd.read_csv(new_tetfp, sep=',',
                     dtype={'代碼': np.int32, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')
    df['中文簡稱'] = df['中文簡稱'].str.split().apply(lambda x: x[0].replace(' ', ''))
    mean_deal = df['成交張數(張)'].mean()
    std_deal = df['成交張數(張)'].std()
    for d_num, s_num in zip(crawlDate, stockNo):

        url = url.replace(old_date, d_num)
        url = url.replace(old_num, s_num)
        old_date = d_num
        old_num = s_num

        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'lxml')
        stock_daily = json.loads(soup.p.text)
        data = stock_daily['data']

        # At new month start, data will insufficient
        if len(data) < timeStep+1:
            dif = (timeStep+1) - len(data)
            print('Need more data, input new date to crawl:', end='')
            d_num = input()
            url = url.replace(old_date, d_num)
            old_date = d_num
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'lxml')
            stock_daily = json.loads(soup.p.text)
            data_supply = stock_daily['data'][-dif:]
            data = data_supply+data

        data = np.array(data[-(timeStep+1):])
        tmp = []
        for i in range(timeStep):
            raw.append([float(data[i][3]), float(data[i][4]), float(data[i][5]), float(data[i][6]), float(data[i][8].replace(',',''))])
            open = float(data[i+1][3])-float(data[i][3])
            high = float(data[i+1][4])-float(data[i][4])
            low = float(data[i+1][5])-float(data[i][5])
            close = float(data[i+1][6])-float(data[i][6])
            deal = (float(data[i+1][8].replace(',',''))-mean_deal)/std_deal-(float(data[i][8].replace(',',''))-mean_deal)/std_deal
            tmp.append([open, high, low, close, deal])
        raw.append([float(data[i+1][3]), float(data[i+1][4]), float(data[i+1][5]), float(data[i+1][6]), float(data[i+1][8].replace(',',''))])
        seq.append(tmp)
    return raw, seq
def append_data(rawData, pred):
    copy = rawData[1:]
    tmp = rawData[1:]
    lastest = tmp[-1]
    new_data = [lastest[0]+pred[0], lastest[1]+pred[1], lastest[2]+pred[2], lastest[3]+pred[3], lastest[4]+pred[4]]
    return copy.append(new_data)

def create_seq(raw, timestep):
    seq = []
    df = pd.read_csv(new_tetfp, sep=',',
                     dtype={'代碼': np.int32, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')
    df['中文簡稱'] = df['中文簡稱'].str.split().apply(lambda x: x[0].replace(' ', ''))
    mean_deal = df['成交張數(張)'].mean()
    std_deal = df['成交張數(張)'].std()
    for i in range(timestep):
        open = raw[i+1][0] - raw[i][0]
        high = raw[i+1][1] - raw[i][1]
        low = raw[i+1][2] - raw[i][2]
        close = raw[i+1][3] - raw[i][3]
        deal = (raw[i+1][4] - mean_deal) / std_deal - (
            raw[i][4] - mean_deal) / std_deal
        tmp = [open, high, low, close, deal]
        seq.append(tmp)
    return seq
def getTarget(stockNo, timeStep, date='20180511'):
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
    mean_vol = np.mean(df['成交張數(張)'])
    std_vol = np.std(df['成交張數(張)'])
    seq = []
    closes = []
    lower_date = df['日期'].unique()[(np.where(df['日期'].unique()==date)[0][0]-timeStep)]
    for stock in stockNo:
        data = df[df['代碼']==stock]
        data_mask = df2[df2['證券名稱']==stock]
        data = data[(data['日期']>=lower_date)&(data['日期']<=date)]
        data_mask = data_mask[(data_mask['日期']>=lower_date)&(data_mask['日期']<=date)]

        tmp = []
        for step in range(timeStep):
            data4 = data_mask.iloc[step+1]
            open = data.iloc[step+1]['開盤價(元)']-data.iloc[step]['開盤價(元)']
            high = data.iloc[step+1]['最高價(元)']-data.iloc[step]['最高價(元)']
            low = data.iloc[step+1]['最低價(元)']-data.iloc[step]['最低價(元)']
            close = data.iloc[step+1]['收盤價(元)']-data.iloc[step]['收盤價(元)']
            vol = ((data.iloc[step+1]['成交張數(張)']-data.iloc[step]['成交張數(張)'])-mean_vol)/std_vol
            tmp.append([open, high, low, close, vol, data4['投信買進股數'], data4['投信賣出股數'], data4['外資買進股數'], data4['外資賣出股數'],
                        data4['自營商買進股數'], data4['自營商賣出股數']])
            if step == timeStep-1:
                closes.append(data.iloc[step+1]['收盤價(元)'])
        seq.append(tmp)
    return seq, closes
def getETF():
    proxies = {
        "http": "47.90.211.250"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome",
        "Accept": "text/html,application/xhtml+xml,application/xml; q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6,zh-CN;q=0.5"
    }
    ua = UserAgent()
    proxy_lists = ['45.6.216.79', '34.213.199.191', '35.225.208.4', '66.119.180.104', '35.188.131.123']
    url = 'http://www.tse.com.tw/fund/T86?response=json&date=20180514&selectType=0099P&_=1526384857982'
    replace_date = '20180514'
    df = pd.read_csv(filename, encoding='big5-hkscs', dtype={'日期': str})
    delta_lists = df['日期'].unique()
    with open('Third_Party.csv', 'a+', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        for num, date in enumerate(delta_lists):
            target_date = date
            print(target_date)
            proxies['http'] = proxy_lists[random.randint(0,4)]
            headers['User-Agent'] = ua.random
            try:
                res = requests.get(url.replace(replace_date, target_date), proxies=proxies, headers=headers)
                stock_daily = json.loads(res.text)
                data = stock_daily['data']
            except Exception as err:
                with open('error.csv', 'a+', encoding='utf-8-sig') as csvfile2:
                    writer2 = csv.writer(csvfile2)
                    writer2.writerow([target_date, err])
                print('Proxy: {}'.format(proxies['http']))
                print('Headers: {}'.format(res.request.headers))
                print(err)
                continue
            print('Proxy: {}'.format(proxies['http']))
            print('Headers: {}'.format(res.request.headers))
            time.sleep(random.randint(5,10))
            for j in range(len(data)):
                for k in range(len(data[j])):
                    data[j][k] = data[j][k].replace(',', '')
                    data[j][k] = data[j][k].replace(' ', '')
                writer.writerow([target_date.replace('-','')]+data[j])
            if num == 0:
                break




if __name__ == '__main__':
    """res = requests.get('http://www.tse.com.tw/fund/T86?response=json&date=20180515&selectType=0099P&_=1526384857982')
    soup = BeautifulSoup(res.text, 'lxml')
    stock_daily = json.loads(soup.p.text)
    data = stock_daily['data']"""
    #getETF()
    """stockNo = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6201',
                     '6203', '6204', '6208', '690', '692', '701', '713']
    seq, closes = getTarget(stockNo, 6)
    df = pd.read_csv(filename, sep=',',
                     dtype={'代碼': np.str, '日期': str, '中文簡稱': str, '開盤價(元)': np.float32, '最高價(元)': np.float32,
                            '最低價(元)': np.float32, '收盤價(元)': np.float32, '成交張數(張)': np.int32}, encoding='big5-hkscs')"""