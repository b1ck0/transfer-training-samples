import os

import tensorflow as tf
import tensorflow_datasets as tfds


def get_data(batch_size=32, shuffle_buffer=1024):
    AUTO = tf.data.experimental.AUTOTUNE
    train, valid = tfds.load('mnist', split=['train', 'test'], shuffle_files=True)

    train = train.shuffle(shuffle_buffer).batch(batch_size).prefetch(AUTO)
    valid = valid.batch(batch_size).prefetch(AUTO)

    def parse_dataset(batch):
        return batch['image'], batch['label']

    train = train.map(parse_dataset, num_parallel_calls=AUTO)
    valid = valid.map(parse_dataset, num_parallel_calls=AUTO)

    return train, valid


def get_model(input_size=(28, 28, 1)):
    inputs = tf.keras.layers.Input(input_size, name='INPUT')
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.15)(x)
    x = tf.image.grayscale_to_rgb(x)
    x = tf.keras.layers.experimental.preprocessing.Resizing(32, 32)(x)

    x = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None)(x)

    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(10, activation='softmax', name='OUTPUT')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # setting up the TF_CONFIG environmental variable for distributed training

    if os.environ['ROLE'] == 'master':
        tf_config = \
            '''{
                "cluster": {
                    "worker": ["10.31.234.86:2222"],
                    "chief": ["10.31.234.148:2222"]
                },
                "task": {"type": "chief", "index": 0}
            }'''
    else:
        tf_config = \
            '''{
                "cluster": {
                    "worker": ["10.31.234.86:2222"],
                    "chief": ["10.31.234.148:2222"]
                },
                "task": {"type": "worker", "index": 0}
            }'''

    os.environ['TF_CONFIG'] = tf_config

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    task_type = strategy.cluster_resolver.task_type
    task_id = strategy.cluster_resolver.task_id

    per_worker_batch = 32
    global_batch_size = strategy.num_replicas_in_sync * per_worker_batch

    with strategy.scope():
        model = get_model(input_size=(28, 28, 1))
        train, valid = get_data(batch_size=global_batch_size, shuffle_buffer=1024)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA

        train = train.with_options(options)

    model.fit(train, validation_data=valid, epochs=10)
