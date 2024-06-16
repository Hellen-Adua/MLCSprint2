import numpy as np
import pandas as pd

class TestDataGenerator:
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of observations
    n_samples = 50
    def __init__(self, data = None) -> None:
        self.data = data
    
    # Function to generate random data for each feature
    def generate_feature(self, mean, std, n_samples):
        return np.random.normal(mean, std, n_samples)
    
    def test_data(self):
    # Generate features
        self.data = {
            'mean_radius': self.generate_feature(14.0, 3.5, self.n_samples),
            'mean_texture': self.generate_feature(19.0, 4.5, self.n_samples),
            'mean_perimeter': self.generate_feature(87.0, 24.0, self.n_samples),
            'mean_area': self.generate_feature(550.0, 330.0, self.n_samples),
            'mean_smoothness': self.generate_feature(0.1, 0.02, self.n_samples),
            'mean_compactness': self.generate_feature(0.1, 0.05, self.n_samples),
            'mean_concavity': self.generate_feature(0.09, 0.08, self.n_samples),
            'mean_concave_points': self.generate_feature(0.05, 0.03, self.n_samples),
            'mean_symmetry': self.generate_feature(0.18, 0.03, self.n_samples),
            'mean_fractal_dimension': self.generate_feature(0.06, 0.01, self.n_samples),
            'radius_se': self.generate_feature(0.4, 0.2, self.n_samples),
            'texture_se': self.generate_feature(1.2, 0.5, self.n_samples),
            'perimeter_se': self.generate_feature(2.8, 1.5, self.n_samples),
            'area_se': self.generate_feature(40.0, 20.0, self.n_samples),
            'smoothness_se': self.generate_feature(0.01, 0.005, self.n_samples),
            'compactness_se': self.generate_feature(0.03, 0.015, self.n_samples),
            'concavity_se': self.generate_feature(0.04, 0.02, self.n_samples),
            'concave_points_se': self.generate_feature(0.01, 0.005, self.n_samples),
            'symmetry_se': self.generate_feature(0.02, 0.01, self.n_samples),
            'fractal_dimension_se': self.generate_feature(0.003, 0.002, self.n_samples),
            'worst_radius': self.generate_feature(16.5, 4.5, self.n_samples),
            'worst_texture': self.generate_feature(25.0, 7.0, self.n_samples),
            'worst_perimeter': self.generate_feature(110.0, 33.0, self.n_samples),
            'worst_area': self.generate_feature(880.0, 590.0, self.n_samples),
            'worst_smoothness': self.generate_feature(0.14, 0.03, self.n_samples),
            'worst_compactness': self.generate_feature(0.25, 0.15, self.n_samples),
            'worst_concavity': self.generate_feature(0.27, 0.20, self.n_samples),
            'worst_concave_points': self.generate_feature(0.13, 0.08, self.n_samples),
            'worst_symmetry': self.generate_feature(0.29, 0.07, self.n_samples),
            'worst_fractal_dimension': self.generate_feature(0.08, 0.02, self.n_samples),
        }

        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        return self.data