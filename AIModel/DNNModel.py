import tensorflow as tf


def dnn_model():
    """
    :return:
        - model : tensorflow model object
    """
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
