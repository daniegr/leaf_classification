import tensorflow as tf
import load_data
import model

tf.logging.set_verbosity(tf.logging.INFO)

# Main method of program
def main(unused_argv):

    # Extract data
    X_train, X_test, y_train, y_test = load_data.load_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Initialize, train and test model

    # Create model from estimator
    regressor = tf.estimator.Estimator(
        model_fn=model.cnn_model_fn, model_dir="tmp/cnn-test")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train model on training set
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        batch_size=2,
        num_epochs=None,
        shuffle=True)
    regressor.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])

    print("Trained model")

    # Test model on test set
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = regressor.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    print("Tested model")

if __name__ == "__main__":
    tf.app.run()