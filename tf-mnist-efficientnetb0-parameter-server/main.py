import os

import tensorflow as tf
import tensorflow_datasets as tfds


def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)

    dataset = tfds.load('mnist', split=['train'])
    dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset.batch(batch_size)
    dataset.prefetch(2)

    return dataset


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

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    # setting up the TF_CONFIG environmental variable for distributed training
    if os.environ['ROLE'] == 'master':
        tf_config = \
            '''{
                "cluster": {
                    "worker": ["10.31.234.86:2222"],
                    "chief": ["10.31.234.148:2222"],
                    "ps": ["10.31.234.171:2222"]
                },
                "task": {"type": "chief", "index": 0}
            }'''
    elif os.environ['ROLE'] == 'worker':
        tf_config = \
            '''{
                "cluster": {
                    "worker": ["10.31.234.86:2222"],
                    "chief": ["10.31.234.148:2222"],
                    "ps": ["10.31.234.171:2222"]
                },
                "task": {"type": "worker", "index": 0}
            }'''
    else:
        tf_config = \
            '''{
                "cluster": {
                    "worker": ["10.31.234.86:2222"],
                    "chief": ["10.31.234.148:2222"],
                    "ps": ["10.31.234.171:2222"]
                },
                "task": {"type": "ps", "index": 0}
            }'''

    os.environ['TF_CONFIG'] = tf_config

    print(os.environ['ROLE'])
    print(os.environ['TF_CONFIG'])

    strategy = tf.distribute.experimental.ParameterServerStrategy()

    with strategy.scope():
        model = get_model(input_size=(28, 28, 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
        steps_per_execution=10
    )

    train = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
    model.fit(train, epochs=10, steps_per_epoch=1875)
