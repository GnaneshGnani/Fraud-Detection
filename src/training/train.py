import os
import torch
import joblib
import pandas as pd
from clearml import Task
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc

from src.models.model import PyTorchModel

param_grid = {
    'hidden_size': [16, 32],
    'lr': [0.001, 0.01],
    'num_epochs': [10, 20]
}

task = Task.init(
    project_name = "Fraud Detection",  # Name of your project in ClearML UI
    task_name = "PyTorch Model Grid Search",  # Name of this experiment
)

def main():
    df = pd.read_csv("data/baseline.csv")

    drop_columns = ["id"]
    df = df.drop(columns = drop_columns)

    min_max_scaler = MinMaxScaler()
    df["Amount"] = min_max_scaler.fit_transform(df["Amount"].values.reshape(-1 ,1))
    joblib.dump(min_max_scaler, 'artifacts/min_max_scaler.pkl')

    X = df.drop(columns = "Class").values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PyTorchModel(input_size = X.shape[1], output_size = 1)

    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print("Test accuracy: ", test_score)
    print(best_model)

    best_params = grid_search.best_params_
    task.connect({
        'learning_rate': best_params['lr'],
        'hidden_size': best_params['hidden_size'],
        'epochs': best_params['num_epochs']
    })

    logger = task.get_logger()

    for mode, X, y in zip(['Train', 'Test'], [X_train, X_test], [y_train, y_test]):
        y_pred = best_model.predict(X)
        y_prob = best_model.predict_proba(X)

        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        
        # Log the metrics to ClearML
        logger.report_scalar('Accuracy', mode, iteration = 0, value = accuracy)
        logger.report_scalar('F1-score', mode, iteration = 0, value = f1)
        logger.report_scalar('Precision', mode, iteration = 0, value = precision)
        logger.report_scalar('Recall', mode, iteration = 0, value = recall)
        logger.report_scalar('ROC_AUC', mode, iteration = 0, value = roc_auc)
        
        # Generate and save the ROC curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc_value = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc_value)
        plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - Train')
        plt.legend(loc = "lower right")

        # Save the ROC curve plot
        roc_curve_image_path = f'roc_curve_{mode.lower()}.png'
        plt.savefig(roc_curve_image_path)
        plt.close()

        task.upload_artifact(name = f'ROC Curve - {mode}', artifact_object = roc_curve_image_path)

    task.upload_artifact(name = "Model", artifact_object = best_model.model)

    model_params = {
        "input_size": X.shape[1],
        "hidden_size": best_params["hidden_size"],
        "output_size": 1
    }

    joblib.dump(model_params, 'artifacts/model_params.pkl')

    curr_models_count = len(os.listdir("artifacts/model"))
    torch.save(best_model.model.state_dict(), "artifacts/model/fraud_model_" + str(curr_models_count + 1) + ".pth")

if __name__ == "__main__":
    main()