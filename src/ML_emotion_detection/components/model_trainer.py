from src.ML_emotion_detection.config.configuration import ModelTrainerConfig
from src.ML_emotion_detection.utils.common import load_bin, save_bin
from sklearn.linear_model import LogisticRegression
import os

class ModelTrainer:
    """
    A class used to train a logistic regression model with provided configuration.

    Attributes:
        config (ModelTrainerConfig): Configuration object containing paths to training data, model parameters, etc.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer with the provided configuration.

        Args:
            config (ModelTrainerConfig): Configuration object containing paths to training data, model parameters, etc.
        """
        self.config = config

    def train(self):
        """
        Trains the logistic regression model using the data specified in the configuration.

        Loads training and test data, trains the model, and saves the trained model to the specified location.
        """
        
        train_x = load_bin(self.config.train_data_path_X)
        test_x = load_bin(self.config.test_data_path_X)
        
        train_y = load_bin(self.config.train_data_path_target)
        test_y = load_bin(self.config.test_data_path_target)

        log_reg = LogisticRegression(class_weight=self.config.class_weight,
                                     max_iter=self.config.max_iter,
                                     penalty=self.config.penalty,
                                     solver=self.config.solver,
                                     n_jobs=self.config.n_jobs,
                                     random_state=self.config.random_state)
        
        log_reg.fit(train_x, train_y)

        save_bin(log_reg, os.path.join(self.config.root_dir, self.config.model_name))
