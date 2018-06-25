import csv
import requests
import numpy as np
import pandas as pd
import json
import time
import random
import os
from fake_useragent import UserAgent

new_file = './data/TBrain_Round2_DataSet_20180608/tetfp.csv'

def getETF(dates):
    root = os.getcwd()
    path = os.path.join(root, 'data')
    path = os.path.join(path, 'Third_party')
    os.chdir(path)
    if not dates:
        return
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
    """
    set range to crawl ! !
    """


    with open('Third_Party2.csv', 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for num, date in enumerate(dates):
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
    os.chdir(root)

def integrateData():
    root = os.getcwd()
    path = os.path.join(root, 'data')
    path = os.path.join(path, 'Third_party')
    os.chdir(path)

    res = requests.get('http://www.tse.com.tw/fund/T86?response=json&date=20171103&selectType=0099P&_=1526384857982')
    fields = ['日期'] + json.loads(res.text)['fields']
    data = []
    with open('Third_Party.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 17:
                data.append(row)
            elif len(row) == 20:
                row[6] = row[3]+row[6]
                row[7] = row[4]+row[7]
                row[8] = row[5]+row[8]
                del row[3]
                del row[3]
                del row[3]
                data.append(row)
    with open('Third_Party2.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 17:
                data.append(row)
            elif len(row) == 20:
                row[6] = row[3]+row[6]
                row[7] = row[4]+row[7]
                row[8] = row[5]+row[8]
                del row[3]
                del row[3]
                del row[3]
                data.append(row)
    with open('Third_Party_Merge.csv', 'a+', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        for row in data:
            writer.writerow(row)

    print('Integrate successfully!')
    os.chdir(root)

def checkDate():
    df = pd.read_csv(new_file, encoding='big5-hkscs', dtype={'日期': str})
    root = os.getcwd()
    path = os.path.join(root, 'data')
    path = os.path.join(path, 'Third_party')
    os.chdir(path)

    check_date = df['日期'].unique()
    current_date = []
    need_date = []
    with open('Third_Party.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if str(row[0]) not in current_date:
                current_date.append(str(row[0]))
    with open('Third_Party2.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if str(row[0]) not in current_date:
                current_date.append(str(row[0]))
    for date in check_date:
        if date not in current_date:
            need_date.append(date)
            print(date)
    os.chdir(root)
    return need_date

def getInput(date):
    df = pd.read_csv('./data/three_big.csv')
    for i in range(4):
        df[df.columns[i + 2]] = df[df.columns[i + 2]].str.split().apply(lambda x: x[0].replace(',', ''))
        df[df.columns[i + 2]] = df[df.columns[i + 2]].astype(np.float64)
    df['日期'] = df['日期'].astype(str)
    names = df['證券名稱'].unique()
    image = []
    seq = []
    date_index = int(np.where(df['日期'].unique()==date)[0][0])
    print('start')
    tmp = []
    data = df[df['日期'] == df['日期'].unique()[date_index-i]]
    for name in names:
        if name not in data['證券名稱'].unique():
            tmp.append([0]*10)
        else:
            target = data[data['證券名稱'] == name]
            target = target.values.tolist()[0][2:]
            tmp.append(target)
    image.append(tmp)
    seq.append(image)
    return seq

if __name__ == '__main__':
    #dates = checkDate()
    #
    # getETF(dates)
    integrateData()
    #seq = getInput('20180518', 6)
