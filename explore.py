import pandas as pd
import numpy as np

class DataExplorer:
    """ takes an input "data" which is the csv file to be analysed
    """
    def __init__(self, data) -> None:
        self.data = data
    
    def print_statistical_summary(self):
        title = "Statistical Summary of dataset"
        summary = self.data.describe().transpose()
        print(title)
        print(summary)

    def print_dataset_information(self):
        title = "A brief information about the data set"
        info = self.data.info()
        print(title)
        print(info)    

    def print_dataframe_shape(self):
        title = "Shape of the data"
        shape = self.data.shape
        print(title)
        print(shape)

    def print_head_of_data(self, size = 5):
        title = "Head of the data"
        head  = self.data.head(size)
        print(title)
        print(head)     

    def print_tail_of_data(self, size =5):     
        title = "Tail of the data"
        tail  = self.data.tail(size)
        print(title)
        print(tail)

    def print_null_values_count(self):
        title = "Count of null values: "
        null_count  = self.data.isnull().sum()
        print(title)
        print(null_count)  

    def print_duplicated_values_count(self):
        title = "Number of duplicated values: "
        duplicates  = self.data.duplicated().sum()
        print(title)
        print(duplicates)

    def print_unique_values_count(self):
        title = "Count of unique values per feature"
        unique_couts  = self.data.nunique()
        print(title)
        print(unique_couts)  

    def print_value_counts(self, category):
        title = "Value counts per category"
        counts = self.data[category].value_counts()
        print(title)
        print(counts)

    def correlation(self):
        pass                


    # print_dataframe_shape
    # print_dataset_information
    # print_duplicated_values_count
    # print_head_of_data
    # print_null_values_count
    # print_statistical_summary
    # print_tail_of_data
    # print_unique_values_count
    # print_value_counts