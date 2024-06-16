import joblib
from supervised_models import TrainTestModels
from process import DataProcessor

class Automation:
    def __init__(self, model) -> None:
        self.model = model

    # def save_model(self, filename):
    #     joblib.dump(self.model, filename)

    # def load_model(self, filename):
    #     return joblib.load(filename)


    def test_new_data(self, data):
        process = DataProcessor(data)
        encoded_data = process.encode_data()
        scaled_data = process.scale_data()

        new_data = scaled_data

        # load trained model
        model = self.model
        predictions = self.model.predict(new_data)
        return predictions
    


