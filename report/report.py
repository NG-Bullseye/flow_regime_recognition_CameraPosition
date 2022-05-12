import tensorflow as tf
import json


if __name__ == '__main__':
    # Load model
    model = tf.keras.models.load_model("training/model/trained_model.h5")
    model.summary()

    # Load data
    tf_dataset_test = tf.data.experimental.load('cache/ds_test.tf')

    # Evaluate model
    model.compile(loss='categorical_crossentropy', metrics='accuracy')
    results = model.evaluate(tf_dataset_test, verbose=False)
    print(f"Test loss {results[0]}, Test acc {results[1]}")

    # Model predict
    y_test = [label for image, label in list(tf_dataset_test.unbatch().as_numpy_iterator())]
    y_test_idx = tf.argmax(y_test, axis=1)
    predictions = model.predict(tf_dataset_test)
    predictions_idx = tf.argmax(predictions, axis=1)

    # Confusion matrix
    conf_mat = tf.math.confusion_matrix(y_test_idx, predictions_idx)
    print(conf_mat)

    # Metrics
    metrics = {'accuracy_test': results[1]}

    with open('metrics.json', 'w') as f:
        f.write(json.dumps(metrics, indent=4))
