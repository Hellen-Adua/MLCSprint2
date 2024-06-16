import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualize:
    def __init__(self, data) -> None:
        self.data = data

    def plot_histogram(self, column, figsize = (10, 6), **kwargs):
        """kwargs for histplot
        sns.set(style='darkgrid')
        sns.set(style='white')

          figsize=(12,8), bins=10, hue='a categorical column', color='red',edgecolor='black',lw=4,ls='--'
          """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.histplot(data=self.data, x=column, **kwargs)
        plt.title('Histogram')
        plt.show()
        

    def plot_distplot(self, column,figsize=(12,8), **kwargs):
        """kwargs for distplot
        sns.set(style='darkgrid')
        sns.set(style='white')

            figsize=(12,8), kde=True, color='red',bins=10, rug=True, edgecolor='black',lw=4,ls='--'
            sns.kdeplot(data=sample_ages,x='age', clip=[0,100], bw_adjust=0.1, shade=True, color='red', 
            linewidth=6)
            """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.histplot(data=self.data, x=column, **kwargs)
        plt.title('Distplot')
        plt.show()    


    def plot_countplot(self, column,figsize=(12,8), **kwargs):
        """kwargs for countplot
            figsize=(12,8), hue='a categorical column', palette='Set1', palette='Paired'

            """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.countplot(data=self.data, x=column, **kwargs)
        plt.title('Countplot')
        plt.show()    

    def plot_barplot(self, x, y, figsize=(12,8), **kwargs):
        """kwargs for barplot
            figsize=(12,8), hue='a categorical column', palette='Set1', palette='Paired'

            """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.barplot(data=self.data, x=x, y=y, estimator=np.mean,ci='sd' **kwargs)
        plt.title('Barplot')
        plt.show()  

    def plot_scatter(self, x, y, figsize=(12,8), **kwargs):

        """kwargs for scatterplot
        figsize=(12,8), hue='diagnosis', size='another categorical column', s=200, palette='viridis', linewidth=0,alpha=0.2, edgecolor='black', marker='o',
        style='another categorical column', legend=False
        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.scatterplot(data=self.data, x=x, y=y, **kwargs)
        plt.title('Scatter Plot')
        plt.show()    

    def plot_box(self, figsize=(12,8), **kwargs):

        """kwargs for boxplot
        figsize=(12,8), x= "a categorical var", y = "a continuous var", orient = "h", width=0.3, hue='a categorical column', palette='Set1', palette='Paired'

        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.boxplot(data=self.data, **kwargs)
        plt.title('Box Plot')
        plt.show()    


    def plot_heatmap(self, figsize=(12,12), **kwargs):
        plt.figure(figsize=figsize)

        kwargs.pop('figsize', None)  # Remove figsize from kwargs

        sns.set_theme(style="white")
        corr = self.data.corr()
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr, mask= mask, cmap=cmap, square=True, fmt=".2f", **kwargs)
        plt.title('Heatmap')
        plt.show() 


    def plot_violin(self, x, y, figsize = (12,8), **kwargs):

        """kwargs for violinplot
        figsize=(12,8), x= "a categorical var", y = "a continuous var", hue='a categorical column', split=True, inner=None, inner='quartile',
        inner='box', inner='stick', palette='Set1', palette='Paired', bw=0.1,

        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.violinplot(data=self.data, x=x, y=y, **kwargs)
        plt.title('Violin Plot')
        plt.show()

    # Comparison plots
    def plot_pairplot(self, figsize= (12,8), **kwargs):
        """ kwargs for pairplort
        figsize=(12,8), hue='a categorical column', palette='Set1', palette='Paired', palette='viridis', corner=True, diag_kind='hist',
        """
        plt.figure(figsize=figsize)
        kwargs.pop('figsize', None)  # Remove figsize from kwargs
        sns.pairplot(self.data,  hue="diagnosis", corner=True,palette='viridis' )
        plt.title('Pairplot')
        plt.show()       