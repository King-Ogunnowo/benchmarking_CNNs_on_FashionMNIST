import tensorflow as tf
from sklearn.model_selection import train_test_split

def process_MNIST_data():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full / 225.0
    X_test = X_test / 225.0
    return X_train_full, y_train_full, X_test, y_test

def extract_validation_set(X_train_full, y_train_full):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, 
        y_train_full, 
        test_size = 0.25, 
        stratify = y_train_full # --- stratify, so we have all classes equally represented. Similar to stratified sampling
    )
    return X_train, X_valid, y_train, y_valid