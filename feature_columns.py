import tensorflow as tf

feature_column = tf.feature_column.numeric_column(key="1")
feature_column = tf.feature_column.numeric_column(key='1', shape=10)
print(feature_column)

bucket_feature = tf.feature_column.numeric_column('2')
bucketized = tf.feature_column.bucketized_column(
    source_column = bucket_feature,
    boundaries = [1960, 1980, 2000])
print (bucketized)

identity_feature = tf.feature_column.categorical_column_with_identity(
    key='3',
    num_buckets=4)
print(identiy_feature)



