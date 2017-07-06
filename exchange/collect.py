import poloniex
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    date, data = getDataPoloniex()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(date, data)
    ax.set_title("BTC Price")
    ax.set_xlabel("Day")
    ax.set_ylabel("BTC Price[$]")
    plt.grid(fig)
    plt.show(fig)

def getDataPoloniex():
    polo = poloniex.Poloniex()
    polo.timeout = 2
    chartETH_BTC = polo.returnChartData('BTC_ETH', period=300, start=time.time() - 300 * 500, end=time.time())

    row_list = []
    for i in range(len(chartETH_BTC)):
        # date = datetime.datetime.fromtimestamp(chartETH_BTC[i]['date']).date()
        item = chartETH_BTC[i]
        dict = [ str(item['date']),  str(item['open']), str(item['high']),  str(item['low']), str(item['close']), str(item['volume'])]
        row_list.append(dict)
    print(row_list)
    df = pd.DataFrame(row_list)
    df.to_csv('polo.csv', index=False, header=False)

    # tmpDate = [chartETH_BTC[i]['date'] for i in range(len(chartETH_BTC))]
    # date = [datetime.datetime.fromtimestamp(tmpDate[i]).date() for i in range(len(tmpDate))]
    # data = [float(chartETH_BTC[i]['open']) for i in range(len(chartETH_BTC))]
    return

if __name__ == "__main__":
    getDataPoloniex()