import tensorflow as tf


class BaseModel(tf.keras.Model):
    def call(self, inputs, training=False):
        return inputs

    def get_stacked_bands(self, inputs, prefix=''):
        return tf.stack([
            inputs[prefix + 'B1'],
            inputs[prefix + 'B2'],
            inputs[prefix + 'B3'],
        ], axis=3)


class ResNet50v2(BaseModel):
    def __init__(self, feature_size, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.resnet_model = tf.keras.applications.ResNet50V2(weights=None, include_top=False,
                                                             input_tensor=tf.keras.Input(shape=(256, 256, 3)))
        self.final_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.final_layer = tf.keras.layers.Dense(feature_size, activation=None)

    def call(self, inputs, training=False):
        inputs = self.get_stacked_bands(inputs)
        x = self.resnet_model(inputs, training=training)
        x = self.final_pool(x)
        return self.final_layer(x)


class SCNN(BaseModel):
    def __init__(self, feature_size, **kwargs):
        super(SCNN, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flat = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(feature_size, activation=None)

    def call(self, inputs, training=False):
        inputs = self.get_stacked_bands(inputs)
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.pool3(x)

        x = self.flat(x)
        return self.final_layer(x)


class TripletModel(BaseModel):
    def __init__(self, model_arch, feature_size, **kwargs):
        super(TripletModel, self).__init__(**kwargs)
        if model_arch == 'resnet':
            self.internal_model = ResNet50v2(feature_size, **kwargs)
        elif model_arch == 'scnn':
            self.internal_model = SCNN(feature_size, **kwargs)
        else:
            raise ValueError(f'Invalid model architecture "{model_arch}"')

    def call(self, inputs, training=False):
        return self.internal_model(inputs, training)

    def predict(self, inputs):
        return self(inputs)
