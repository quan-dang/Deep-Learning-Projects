from utils import *
from model import *

def main():
    print("Loading the data")
    processed_data = load_and_preprocess_data()

    print("Getting train and test dataset")
    X_train, X_test, y_test = get_train_and_test_data(processed_data)

    model_obj = MODEL(X_train, X_test, y_test)

    print("Training model")
    model_obj.train_model()
    
    print("Loading the train model")
    model_obj.get_trained_model()
    
    print("Get reconstruction loss by class")
    model_obj.plot_reconstruction_error_by_class()

    print("Getting precision recall curves by thresholds")
    model_obj.get_precision_recall_curves()

    print("Get confusion matrix with 80% recall on test dataset")
    model_obj.get_confusion_matrix(min_recall=0.8)

if __name__== main():
    main()
