import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class TrainTestModels:
    def  __init__(self) -> None:
        pass

    def train_and_test_model(self, model, X, y):
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train the model on the training data
        model.fit(x_train, y_train)
        training_accuracy = f'Accuracy on training {model}:', model.score(x_train, y_train) * 100

        # Print the accuracy on the test data
        testing_accuracy = f'Accuracy on testing {model} :', model.score(x_test, y_test) * 100

        # Generate predictions on the test data
        y_pred = model.predict(x_test)

        # Print the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        # print(report.split())

        return model, report, training_accuracy, testing_accuracy
    
    def extract_metrics(self, report_dict):
        metrics_of_interest = ['precision', 'recall', 'f1-score']

        extracted_values = {metric: {} for metric in metrics_of_interest}
        for model, report in report_dict.items():
            for metric in metrics_of_interest:
                if 'weighted avg' in report:
                    extracted_values[metric][model] = report['weighted avg'][metric]
        return extracted_values
    
    def plot_scores(self, model_scores, model_scores_after_pca):
        # Extract metrics for both conditions
        before_pca_metrics = self.extract_metrics(model_scores)
        after_pca_metrics = self.extract_metrics(model_scores_after_pca)

        # Create DataFrames for each condition
        df_before_pca = pd.DataFrame(before_pca_metrics).transpose()
        df_after_pca = pd.DataFrame(after_pca_metrics).transpose()

        # Rename columns to indicate the condition
        df_before_pca.columns = [f'{col}_Before_PCA' for col in df_before_pca.columns]
        df_after_pca.columns = [f'{col}_After_PCA' for col in df_after_pca.columns]

        # Combine the DataFrames
        df_combined = pd.concat([df_before_pca, df_after_pca], axis=1)

        # Prepare the DataFrame for plotting
        df_plot = df_combined.reset_index().melt(id_vars='index', var_name='Condition', value_name='Score')
        df_plot[['Model', 'Condition']] = df_plot['Condition'].str.split('_', n=1, expand=True)
        print(df_plot.head())
        print(df_plot.columns)

        # Plotting for all models
        plt.figure(figsize=(12, 6))
        sns.barplot(x='index', y='Score', hue='Condition', data=df_plot, palette="viridis")
        plt.title('Model Performance Metrics for all modelsBefore and After PCA')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.ylim(0, 1)  # Assuming scores are between 0 and 1
        plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.show()

        # Plotting score for each model
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='Score', hue='Condition', data=df_plot, palette="viridis")
        plt.title('Model Performance Metrics Before and After PCA')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.ylim(0, 1)  # Assuming scores are between 0 and 1
        plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.show()    

        # Catplot for all models, for all metrics
        n_models = len(df_plot['Model'].unique())
        n_cols = 2  # Number of columns you want

        # Plotting
        g = sns.catplot(
            data=df_plot,
            x='index', y='Score', hue='Condition',
            col='Model', kind='bar',
            palette="viridis", height=6, aspect=1.2,
            col_wrap=n_cols
        )
        g.fig.suptitle('Model Performance Metrics Before and After PCA - all models, all metrics', y=1.03)
        g.set_axis_labels("Metric", "Score")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1))
        g.add_legend(title='Condition', loc="best")

        plt.show()