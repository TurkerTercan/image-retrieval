import tensorflow as tf


def get_feature_desc():
    return {
        'B1': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B2': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'B3': tf.io.FixedLenFeature([256 * 256], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'patch_name': tf.io.VarLenFeature(dtype=tf.string)
    }


def transform_example(example):
    return (
        {
            'B1': tf.reshape(example['B1'], [256, 256]),
            'B2': tf.reshape(example['B2'], [256, 256]),
            'B3': tf.reshape(example['B3'], [256, 256]),
            'patch_name': tf.sparse.to_dense(example['patch_name']),
        },
        {
            'label': tf.cast(example['label'], tf.float32),
        }
    )


def load_archive(filename, batch_size=0, shuffle_size=0, num_parallel_calls=10, prefetch_size=0, shuffle_seed=None):
    feature_desc = get_feature_desc()
    dataset = tf.data.TFRecordDataset(filename)
    if shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size, seed=shuffle_seed)

    def parse_example(example):
        return transform_example(tf.io.parse_single_example(example, feature_desc))

    dataset = dataset.map(parse_example, num_parallel_calls)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset
