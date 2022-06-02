'''temp = {0: 299.7, 1: 300.0, 2: 299.6, 3: 299.3, 4: 299.3, 5: 299.1, 6: 299.2, 7: 298.9, 8: 298.8, 9: 298.0, 10: 298.1, 11: 298.6, 12: 298.6, 13: 298.7, 14: 298.6, 15: 298.7, 16: 299.1, 17: 298.3, 18: 298.3, 19: 298.7, 20: 298.6, 21: 298.6, 22: 298.1, 23: 298.3, 24: 297.2, 25: 297.2, 26: 297.2, 27: 297.3, 28: 297.0, 29: 297.1, 30: 296.8, 31: 296.5, 32: 296.2, 33: 296.4, 34: 296.5, 35: 296.6, 36: 296.7, 37: 296.7, 38: 296.6, 39: 297.0, 40: 296.7, 41: 296.5, 42: 296.6, 43: 296.4, 44: 296.5, 45: 296.4, 46: 296.2, 47: 296.2, 48: 296.4, 49: 296.2, 50: 296.2, 51: 296.2, 52: 296.2, 53: 296.1, 54: 295.9, 55: 295.8, 56: 296.1, 57: 295.8, 58: 295.8, 59: 295.8, 'target': 296.9}

with open(r"ML\adidas_01_05.csv", "w") as file:

     file.write("minute,price\n")
     for key, value in temp.items():
          string = f"{key},{value}\n"
          file.write(string)'''

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)

df = pd.read_csv(r"C:\Users\Lennard\Desktop\studium\Master_Wirtschaftsinformatik\Projektseminar\l2-backtest-engine-v2-main\efn2_backtesting\book\Book_Adidas_DE_20210101_20210101.csv.gz", compression="gzip")
if df.empty:
    pass
else:
    df.drop(df.columns[2:], axis=1, inplace=True)
    df.drop(df.tail(120000).index, inplace=True)
    df["TIMESTAMP_UTC"] = df["TIMESTAMP_UTC"].astype("string")
    df["TIMESTAMP_UTC"] = df["TIMESTAMP_UTC"].str.slice(start=11, stop=16)
    df.drop_duplicates(subset=["TIMESTAMP_UTC"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    #df["TIMESTAMP_UTC"].map(lambda x: df["TIMESTAMP_UTC"][:18], inplace=True)

    df.drop(axis=0, index=[x for x in range(60, 120)], inplace=True)
    df.drop(df.tail(30).index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    #todo write reduced df to csv
    df.to_csv(r"ML\adidas_01_01")

    with open(r"ML\adidas_01_05.csv", "w") as file:
        file.write("minute,price\n")
        for key, value in temp.items():
            string = f"{key},{value}\n"
            file.write(string)
            '''
    

print(df)
print(len(df))


