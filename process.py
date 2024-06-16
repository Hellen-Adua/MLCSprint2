import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    def __init__(self, data) -> None:
        self.data = data

    def encode_data(self):
        # ''' a is one vlaue of the categorical column to be encoded as 1, the other is encoded as 0'''
        # self.data["target"] = self.data[target].map(lambda row: 1 if row == a else 0 )
        # return self.data 
        # Encode categorical columns
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = LabelEncoder().fit_transform(self.data[col])
        self.encoded = True  # Set the flag to True after encoding
        return self.data   
    
    def scale_data(self):
        
       # Check if encoding has been done
        if not self.encoded:
            raise ValueError("Data must be encoded before scaling. Call encode() first.")
        
        # Scale numerical columns
        scaler = StandardScaler()
        self.data[self.data.columns] = scaler.fit_transform(self.data)
        return self.data