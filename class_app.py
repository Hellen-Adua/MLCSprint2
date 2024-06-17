# import primary modules 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
import joblib

# import secondary packages
from explore import DataExplorer
from viz import Visualize
from process import DataProcessor
from dreducer import DimensionalityReducer
from supervised_models import TrainTestModels
from automate_model import Automation
from test_data import TestDataGenerator

# load data
csv_file_path = "data.csv"
df = pd.read_csv(csv_file_path)

# three smaller datasets for pairlots
columns = list(df.columns)

# create a list of all the columns of possible combinations
mean_columns = [col for col in columns if 'mean' in col]
se_columns = [col for col in columns if 'se' in col]
worst_columns = [col for col in columns if 'worst' in col]

# add the diagnosis column
mean_columns.append("diagnosis")
se_columns.append("diagnosis")
worst_columns.append("diagnosis")

# create the data frames
df_mean = df[mean_columns]
df_se = df[se_columns]
df_worst = df[worst_columns]



class Application:
    def __init__(self) -> None:

        self.process_data = DataProcessor(df)

        self.encoded = self.process_data.encode_data()
        self.d_reducer = DimensionalityReducer(self.encoded, n_components=2) 
        self.trained_models = {}
        self.model_scores = {}
        self.test_accuracy = []
        self.train_accuracy = []

        self.trained_models_after_pca = {}
        self.model_scores_after_pca = {}
        self.test_accuracy_after_pca = []
        self.train_accuracy_after_pca = []

                # Create a list of all models
        self.models = {
                'svc' : SVC(C=3),
                'Naive Bayes': MultinomialNB(),
                'Logistic Regression': LogisticRegression(max_iter=10000),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier()
            }
        self.trainer = TrainTestModels()
            

    def introduction(self):
        text = """# **Breast Cancer Wiscnosin Dataset Description**

        Features were computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
        They describe characteristics of the cell nuclei present in the image.

        Attribute Information:

        1. ID number
        2. Diagnosis (M = malignant, B = benign)
        3. The remaining columns 3-32:

        Ten real-valued features are computed for each cell nucleus:

        * radius (mean of distances from center to points on the perimeter)
        * texture (standard deviation of gray-scale values)
        * perimeter
        * area
        * smoothness (local variation in radius lengths)
        * compactness (perimeter^2 / area - 1.0)
        * concavity (severity of concave portions of the contour)
        * concave points (number of concave portions of the contour)
        * symmetry
        * fractal dimension ("coastline approximation" - 1)

        The mean, standard error and "worst" or largest (mean of the three largest values) of these 
        features were computed for each image, resulting in 30 features.

        * For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

        All feature values are recoded with four significant digits.

        Missing attribute values: none

        Class distribution: 357 benign, 212 malignant"""
        return text
    def import_modules(self):
        text = '''The imported modules are Imported modules are \n
            import streamlit as st\n
            import pandas as pd\n
            import numpy as np\n
            import matplotlib.pyplot as plt\n
            import seaborn as sns\n
            from sklearn.model_selection import train_test_split\n
            from sklearn.linear_model import LogisticRegression\n
            from sklearn.tree import DecisionTreeClassifier\n
            from sklearn.ensemble import RandomForestClassifier\n
            from sklearn.neighbors import KNeighborsClassifier\n
            from sklearn.svm import SVC\n
            from sklearn.preprocessing import StandardScaler, OneHotEncoder\n
            from sklearn.impute import SimpleImputer\n
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n
            from sklearn.pipeline import Pipeline\n
            from sklearn.decomposition import PCA\n
            from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline '''
        return text
        
    def load_data(self):
        text = 'data loaded with pd.read_csv("data.csv")'    
        return text
    
    def explore_data(self):
       # exlpore the data
        st.title("Data Exploration")
        st.write("""Checking the overall structure of the dataset and what each column represents.""")
        explore_data = DataExplorer(df)
        explore_data.print_dataframe_shape()
        explore_data.print_dataset_information()
        explore_data.print_head_of_data()
        explore_data.print_statistical_summary()
        explore_data.print_null_values_count()
        explore_data.print_duplicated_values_count()

    def basic_visualisation(self):
        print("Data distribution as shown by countplot, boxplots")
        # visualize the data
        data_viz = Visualize(df)
        data_viz.plot_countplot(column = "diagnosis", palette="viridis")

        data_viz.plot_box()

    def visualisation_with_pairplots(self):
        # instantiate the mean visualizer
        data_viz = Visualize(df_mean)
        data_viz.plot_pairplot(figsize=(15,15))

        # instantiate the standard error visualizer
        data_viz = Visualize(df_se)
        data_viz.plot_pairplot(figsize=(15,15))

        # instantiate the worst dimension visualizer
        data_viz = Visualize(df_worst)
        data_viz.plot_pairplot(figsize=(15,15))

    def heat_map(self):
        st.title("Heat Map")
        vis = Visualize(self.encoded)
        vis.plot_heatmap()
        st.markdown("""
        * In a correlation heat map, the higher the correlation value, the more correlated the two variables are:
        
        * Radius, Area and Perimeter are correlated (corr>0.9) which is obvious as area and perimeter is calculated using the radius values.
                
        * Texture_mean and texture_worst are higly correlated with corr_value = 0.98 (texture_worst is the largest value of all the textures).
                
        * Compactness_mean,concavity_mean,concave_points_mean are also highy correlated with values in range 0.7 to 0.9.
                
        * Symmetry_mean and symmetry_worst are correlated too by values 0.7.
                
        * Fractural_dimension_mean and fractural_dimension_worst are correlated by value 0.77
        """)

    def  preprocessing_and_feature_engineering(self):
        st.markdown("### Performed dimensionality redcution using PCA")
        self.d_reducer = DimensionalityReducer(self.encoded, n_components=2)
        # d_reducer.apply_pca()
        self.d_reducer.plot_pca(categorical_value= self.encoded["diagnosis"])
        self.d_reducer.plot_scree_plot()        
        return self.d_reducer

    def training_and_testing(self):
       
        # Train and test each model, and store the structured classification report
        # train model before PCA
        st.write("""The folowing models were trained before dimensionality reduction:
        * SVC 
        * Multinomial Naive Bayes classifier
        * Decision Trees
        * Random Forest Classifier
        * Logistic Regression        
        """)

        for model_name, model in self.models.items():
            model, report, training_accuracy, testing_accuracy= self.trainer.train_and_test_model(model=model, y=self.encoded["diagnosis"], X=self.encoded.drop(["id" ], axis = 1))
            self.trained_models[model_name] = model
            automate = Automation(model)
            automate.save_model(f"{model_name}.pkl")
            self.model_scores[model_name] = report
            self.train_accuracy.append(training_accuracy)
            self.test_accuracy.append(testing_accuracy)
            st.write(training_accuracy)
            st.write(testing_accuracy)
            

    def train_after_pca(self):
        st.write("""The folowing models were trained after dimensionality reduction with PCA:
            * SVC 
            * Decision Trees
            * Random Forest Classifier
            * Logistic Regression        
            """)
        # Negative values in data cannot be passed to MultinomialNB (input X), so we drop it
        x = self.d_reducer.apply_pca()
        y = self.encoded["diagnosis"]

        for model_name, model in self.models.items():
            if model_name != 'Naive Bayes':
                model, report, training_accuracy, testing_accuracy= self.trainer.train_and_test_model(model, x, y)
                self.trained_models_after_pca[model_name] = model
                automate = Automation(model)
                automate.save_model(f"{model_name}_after_pca.pkl")
                self.model_scores_after_pca[model_name] = report
                self.train_accuracy_after_pca.append(training_accuracy)
                self.test_accuracy_after_pca.append(testing_accuracy)
                st.write(training_accuracy)
                st.write(testing_accuracy)
        
                
        # plot metrics for models
        # self.training_and_testing()
            
        extracted_values = self.trainer.extract_metrics(self.model_scores)  

        extracted_values_after_pca = self.trainer.extract_metrics(self.model_scores_after_pca)

        self.trainer.plot_scores(model_scores=self.model_scores, model_scores_after_pca=self.model_scores_after_pca)
        
        return self.trained_models, self.model_scores, self.test_accuracy, self.train_accuracy
        return self.trained_models_after_pca, self.model_scores_after_pca, self.test_accuracy_after_pca, self.train_accuracy_after_pca


    def model_evaluation(self):

        
        st.write(self.test_accuracy)
        st.write(self.train_accuracy)
        st.write(self.test_accuracy_after_pca)
        st.write(self.train_accuracy_after_pca)


    def  test_with_new(self):
        new_data = TestDataGenerator()
        new_df = new_data.test_data()

        model_names = ['svc', 'Logistic Regression', 'Decision Tree', 'Random Forest']

        to_test = st.text_input(f"Name of model to test new data on: {model_names} ")
        if to_test in model_names:
            for model_name, model in self.trained_models_after_pca.items():
                if to_test == model:
                    automate = Automation(model)
                    predictions = automate.test_new_data(new_df)
                    return predictions


    def app(self):
        # Deployment on streamlit
    
        # Prompts for each section
        section_prompts = [
            "Introduction",
            "Import Modules",
            "Load the Data",
            "Exploratory Data Analysis",
            "Visualize with Pairplots",
            "Data Visualisation",
            "Heat Maps",
            "Dimensionality Reduction with Principal Component analysis",
            "Model Training ",
            "Model Training After PCA",
            "Model Evaluation",
            "Summary",
            "Test with new data"
          
        ]
        
        # Sidebar for section selection
        section = st.sidebar.selectbox("Choose a section", section_prompts)
        
        
        # intro = self.introduction()
        # imports = self.import_modules()
        # loading = self.load_data()
        # explore = self.explore_data()
        # train = app.training_and_testing()
        # train_after_pca = self.train_after_pca()
        # evaluate = self.model_evaluation()
        # test_anew = self.test_with_new()
        
        
        if section == "Introduction":
            st.markdown(self.introduction())
        
        elif section == "Import Modules":
            st.write(self.import_modules())
        
        elif section == "Load the Data":
            st.write(self.load_data())
        
        elif section == "Exploratory Data Analysis":
           self.explore_data()
        
        elif section == "Visualize with Pairplots":
            st.write(app.visualisation_with_pairplots())
        
        elif section == "Data Visualisation":
            st.write(app.basic_visualisation())
        
        elif section == "Heat Maps":
            app.heat_map()    
        
        elif section == "Dimensionality Reduction with Principal Component analysis":
            app.preprocessing_and_feature_engineering()
        
        elif section == "Model Training ":
            self.training_and_testing()
        
        elif section == "Model Training After PCA":
            self.training_and_testing()
            self.train_after_pca()
        
        elif section == "Model Evaluation":
            self.training_and_testing()
            self.train_after_pca()
            self.model_evaluation()
        
        elif section == "Test with new data":
            self.test_with_new()
        
        elif section == "Summary":
            st.title("Summary")
            st.markdown('# Noted overall better model perfomance with dimensionality reduction')
            
app = Application()
app.app()

st.write("THE END")


