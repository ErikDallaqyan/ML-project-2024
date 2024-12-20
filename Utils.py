import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score

class DataProccesor:
    def __init__(self, data):
        self.data = data 

    def __str__(self):
        return "Class DataProccesor from Utils.py"
    
    def manual_cross_validation(self, cv_splits_number: int):
        x = self.data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].to_numpy()
        y = self.data['target'].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
        
        logistic_model = LogisticRegression()
        logistic_scores = cross_val_score(logistic_model, x_train, y_train, cv=cv_splits_number, scoring="accuracy")
        logistic_best_score = logistic_scores.mean()

        knn_model = KNeighborsClassifier()
        knn_scores = cross_val_score(knn_model, x_train, y_train, cv=cv_splits_number, scoring="accuracy")
        knn_best_score = knn_scores.mean()

        tree_model = DecisionTreeClassifier()
        tree_scores = cross_val_score(tree_model, x_train, y_train, cv=cv_splits_number, scoring="accuracy")
        tree_best_score = tree_scores.mean()

        best_score = max(logistic_best_score, knn_best_score, tree_best_score)

        if best_score == logistic_best_score:
            print("Best model is Logistic Regression")
        elif best_score == knn_best_score:
            print("Best model is KNN")
        else:
            print("Best model is Decision Tree")

    
    def calculate_metrics(self):
        x = self.data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].to_numpy()
        y = self.data['target'].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = LogisticRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")

    def linear_regression(self):
        x = self.data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].to_numpy()
        y = self.data['target'].to_numpy()
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared_error: {mse}")
        print(f"r2 score: {r2}")


    def k_Nearest_Neighbors(self, max_k_value: int):
        x = self.data
        y = self.data["target"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
        k_values = range(1, max_k_value)
        accuracy_list = []

        for k in k_values:
            model = KNeighborsClassifier(k)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            accuracy_list.append(accuracy)

        plt.plot(k_values, accuracy_list, marker='o')
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("KNN Accuracy for Different k Values")
        plt.show()
    
    def decision_tree(self):
        target = self.data["target"]
        
        x = self.data.to_numpy()
        y = target.to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy is: {accuracy*100}%")

        plot_tree(model, filled=True, class_names=['No Disease', 'Disease'], rounded=True)
        plt.title("Decision Tree for Heart Disease Prediction")
        plt.show()


    def whitespace_check(self):
        rows_to_drop = []

        for idx in self.data.index.to_list():
            row = self.data.iloc[idx]

            if row.isnull().any() or (row == "").any():
                rows_to_drop.append(idx)
        
        if rows_to_drop:
            self.data.drop(index=rows_to_drop, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            return f"Deleted rows with NaN or empty string at positions {rows_to_drop}"
        return "No rows were deleted."

    def normalization(self):
        np_data = self.to_numpy()
        model = MinMaxScaler()

        normalized_data = model.fit_transform(np_data)
        
        return f"Normalized values (first 5 rows): {normalized_data[: 5]}"
     
    def standardization(self):
        np_data = self.data.to_numpy()
        model = StandardScaler()

        standardised_data = model.fit_transform(np_data)

        return f"Standardised values (first 5 rows): {standardised_data[: 5]}"
    
    def add_choresterol_risk(self):
        age = self.data["age"].to_numpy()
        cholesterol = self.data["chol"].to_numpy()
        sex = self.data["sex"].to_numpy()
        #According to ASCVD for every 1 mg/dL in cholesterol, the risk of heart disease could increase by around 2% or 3%
        age_cholesterol_risk = (age * 0.2) + (cholesterol * 0.02) #In real life, more accurate coefficients are used e.g. ASCVD (Atherosclerotic Cardiovascular Disease Risk)

        for i in range(len(age)):
            if age[i] < 30:
                age_cholesterol_risk[i] *= 1.05
            elif 30 <= age[i] < 50:
                age_cholesterol_risk[i] *= 1.1 
            elif 50 <= age[i] < 70:
                age_cholesterol_risk[i] *= 1.2 
            else:
                age_cholesterol_risk[i] *= 1.3

        for j in range(len(sex)):
            if sex[j] == 1: #This is male, if sex is male we need to increase risk by 20%.in real life the risk of getting sick in men is APPROXIMATELY 20% 
                age_cholesterol_risk[j] *= 1.2
        #no change for female

        self.data["Chol risk"] = age_cholesterol_risk # The final result may differ because I did not take into account factors such as Blood pressure, Smoking status, Diabetes

        return f"Age Cholesterol risk successfully added to data"