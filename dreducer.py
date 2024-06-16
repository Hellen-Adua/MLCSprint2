import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

class DimensionalityReducer:    

    def __init__(self, data, n_components, target=None):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.n_components = n_components
        self.target = target
        self.pca = PCA(n_components=self.n_components)
        self.pca_result = self.pca.fit_transform(self.data)

    def apply_pca(self):

        pca_df = pd.DataFrame(data=self.pca_result, columns=[f'PC{i+1}' for i in range(self.n_components)])
        explained_variance = f'Explained variance by components: {self.pca.explained_variance_ratio_}'
        return pca_df

    def plot_pca(self, categorical_value = None):
        pca_df = self.apply_pca()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, c= categorical_value)
        plt.title('PCA Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
        st.pyplot(plt)

    def plot_scree_plot(self):
        # Create a scree plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), self.pca.explained_variance_ratio_, alpha=0.6, color='b')
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), self.pca.explained_variance_ratio_, 'o-', color='b')
        plt.title('Scree Plot')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(self.pca.explained_variance_ratio_) + 1))
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
