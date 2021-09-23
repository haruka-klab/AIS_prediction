import tensorflow as tf
import numpy as np
import pandas as pd

"""
    dataConverter
    データを訓練データと、テストデータの形に変換する(予定)
"""
class dataConverter():
    def __init__(self, file):
        self.df = pd.read_csv(file)
    
    def convert(self):
        self.df = self.df.loc[:,["MMSI","BaseDateTime","LAT","LON","SOG","COG","Heading"]]
        return self.df
    
    def showDf(self):
        print(self.df.head(5))

if __name__=="__main__":
    daC = dataConverter("data/AIS_2019_01_04.csv")
    daC.showDf()
    daC.convert()
    daC.showDf()