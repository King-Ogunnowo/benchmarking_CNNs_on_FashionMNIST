import seaborn as sns
import matplotlib.pyplot as plt

def plot_performance(peformance, file_name):
    plt.figure(figsize = (15, 10))
    sns.lineplot(peformance.history)
    plt.savefig(f"{file_name}.png")
    
def train(model, file_name):
    X_train, X_valid, X_test, y_train, y_valid, y_test = process_MNIST_data()
    history = model.fit(
        X_train, 
        y_train, 
        epochs = 10, 
        validation_data = (
            X_valid, 
            y_valid
        )
    )
    model.save(f"models/{file_name}")
    return history
