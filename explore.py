import pandas as pd
import numpy as np
import streamlit as st
import io

class DataExplorer:
    """ takes an input "data" which is the csv file to be analysed
    """
    def __init__(self, data) -> None:
        self.data = data
    
    def print_statistical_summary(self):
        title = "Statistical Summary of dataset"
        summary = self.data.describe().transpose()
        st.title(title)
        st.write(summary)

    def print_dataset_information(self):
        title = "A brief information about the data set"
        
        # info_df = self.data.info(verbose=False, memory_usage='deep')

        # Convert the info DataFrame to a table format
        
        # info_table = info_df.reset_index()
        # info_table.columns = ['Column', 'Non-Null Count', 'Dtype']
        # info_table['Column'] = info_table['Column'].str.replace(' ', '')

        # Display the info table using st.table()
        # st.write(info_df)
        
        # buffer = io.StringIO()
        # self.data.info(buf=buffer)
        # info_str = buffer.getvalue()
        # return info_str
        # # info = self.data.info()
        st.title(title)
        # # return self.data.info()
        
        # Capture df.info() output as a string
        info_str = io.StringIO()
        self.data.info(buf=info_str)
        info_str = info_str.getvalue()
        
        # Extract information from the string and create a DataFrame
        info_data = []
        for line in info_str.split('\n'):
            if line.strip():
                info_data.append(line.split(':'))
        
        info_df = pd.DataFrame(info_data, columns=['Info', 'Value'])
        
        # Display the info DataFrame using st.table()
        st.table(info_df)
    

    def print_dataframe_shape(self):
        title = "Shape of the data"
        shape = self.data.shape
        st.title(title)
        print(shape)

    def print_head_of_data(self, size = 5):
        title = "Head of the data"
        head  = self.data.head(size)
        st.title(title)
        print(head)     

    def print_tail_of_data(self, size =5):     
        title = "Tail of the data"
        tail  = self.data.tail(size)
        print(title)
        print(tail)

    def print_null_values_count(self):
        title = "Count of null values: "
        null_count  = self.data.isnull().sum()
        st.title(title)
        print(null_count)  

    def print_duplicated_values_count(self):
        title = "Number of duplicated values: "
        duplicates  = self.data.duplicated().sum()
        st.title(title)
        print(duplicates)

    def print_unique_values_count(self):
        title = "Count of unique values per feature"
        unique_couts  = self.data.nunique()
        st.title(title)
        print(unique_couts)  

    def print_value_counts(self, category):
        title = "Value counts per category"
        counts = self.data[category].value_counts()
        st.title(title)
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
