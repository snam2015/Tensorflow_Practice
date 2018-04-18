import tensorflow as tf
import pandas as pd
import argparse

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int,
                    help='number of training steps')

args = parser.parse_args(argv[1:])

def load_data(label_name='Species'):
    train_path= tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1], 
                                        origin=TRAIN_URL)
    train = pd.read_csv(filepath_or_buffer=train_path, 
                        names=CSV_COLUMN_NAMES,
                        header=0
                        )

    train_features, train_label = train, train.pop(label_name)
    
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

def train_

(train_feature, train_label), (test_feature, test_label) = load_data()

my_feature_columns = []

for key in train_feature.keys():
    my_feature_columsn.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNCLassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10,10],
    n_classes=3)

classifier.train(
    input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
    steps= args.train_steps)


