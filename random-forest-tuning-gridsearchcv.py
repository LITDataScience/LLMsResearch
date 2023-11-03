import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


class HeartDiseaseModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_Grid = None

    def preprocess_data(self):
        # Perform one-hot encoding for specified columns
        self.data = pd.get_dummies(self.data, columns=[
            "sex", "chest_pain_type", "fasting_blood_sugar",
            "rest_ecg", "exercise_induced_angina", "slope",
            "vessels_colored_by_flourosopy", "thalassemia"
        ], drop_first=True)

        # Get Target data
        self.y = self.data['target']
        # Load X Variables into a Pandas DataFrame with columns
        self.X = self.data.drop(['target'], axis=1)

    def split_data(self, test_size=0.25, random_state=101):
        # Divide Data into Train and Test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def grid_search(self):
        # Define hyperparameters for the Random Forest model
        n_estimators = [50, 100, 200]
        max_features = ['auto', 'sqrt']
        max_depth = [5, 10, 15]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        # Create the param grid
        param_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }

        # Create and fit the GridSearchCV model
        rf_Model = RandomForestClassifier(random_state=42)
        self.rf_Grid = GridSearchCV(
            estimator=rf_Model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1
        )
        self.rf_Grid.fit(self.X_train, self.y_train)

    def display_best_params(self):
        print("Best Hyperparameters:")
        print(self.rf_Grid.best_params_)

    def display_accuracy(self):
        train_accuracy = self.rf_Grid.score(self.X_train, self.y_train)
        test_accuracy = self.rf_Grid.score(self.X_test, self.y_test)

        print(f'Train Accuracy: {train_accuracy:.3f}')
        print(f'Test Accuracy: {test_accuracy:.3f}')


if __name__ == "__main__":
    data_path = 'C:\\Users\\Rajeev\\PycharmProjects\\LLMsResearch\\data\\HeartDiseaseTrain-Test.csv'

    model = HeartDiseaseModel(data_path)
    model.preprocess_data()
    model.split_data()
    model.grid_search()

    print("\nResults:")
    model.display_best_params()
    model.display_accuracy()


"""
Results:
Best Hyperparameters:
{
    'bootstrap': True,
    'max_depth': 15,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 50
}
Train Accuracy: 1.000
Test Accuracy: 1.000

=======================================================================================================
1. Adjusted the hyperparameter search space: Modified the ranges for hyperparameters like n_estimators,
max_depth, and min_samples_split to include values that might yield better results.

2. Set random_state for the RandomForestClassifier to ensure reproducibility.

3. Increased the number of cross-validation folds (cv=5) for a more robust evaluation.

4. Used all available CPU cores (n_jobs=-1) for parallel processing during GridSearchCV.
=======================================================================================================
"""