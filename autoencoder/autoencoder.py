import tensorflow as tf


def create_blstm_model(fingerprint_input, model_settings, is_training=False):
    time_length = model_settings['spectrogram_length']
    feature_length = model_settings['feature_length']

    units = [240, 80]

    if is_training:
        dropout_prob = tf.compat.v1.placeholder(tf.float32, [])

    layer = tf.keras.layers.Reshape([time_length, feature_length])(fingerprint_input)
    print(layer)

    layer = tf.keras.layers.Dense(feature_length, 'tanh')(layer)
    print(layer)

    for unit in units:
        layer = tf.keras.layers.LSTM(unit, return_sequences=True)(layer)
        print(layer)

    layer = tf.keras.layers.LSTM(40, return_sequences=True)(layer)
    print(layer)

    encoded = tf.keras.layers.Flatten(name='encoded')(layer)
    print(encoded)

    for unit in units[::-1] + [feature_length]:
        layer = tf.keras.layers.LSTM(unit, return_sequences=True)(layer)
        print(layer)

    layer = tf.keras.layers.Dense(feature_length, 'sigmoid')(layer)
    print(layer)

    decoded = tf.keras.layers.Flatten(name='decoded')(layer)
    print(decoded)

    if is_training:
        return decoded, dropout_prob

    return decoded


def create_lstm_model(fingerprint_input, model_settings, is_training=False):
    time_length = model_settings['spectrogram_length']
    feature_length = model_settings['feature_length']
    new_features = [40]

    units = [80]

    if is_training:
        drop_prob = tf.compat.v1.placeholder(tf.float32, [])

    layer = tf.keras.layers.Reshape([time_length, feature_length])(fingerprint_input)
    print(layer)

    layer = tf.keras.layers.Dense(20)(layer)
    print(layer)

    for unit, new_feature in zip(units, new_features):
        layer = tf.keras.layers.LSTM(unit, return_sequences=True)(layer)
        print(layer)

        if is_training:
            layer = tf.keras.layers.Dropout(drop_prob)(layer)

        layer = tf.keras.layers.Dense(new_feature)(layer)
        print(layer)

        if is_training:
            layer = tf.keras.layers.Dropout(drop_prob)(layer)

    encoded = tf.keras.layers.Flatten(name='encoded')(layer)
    print(encoded)

    for unit, new_feature in zip(units[::-1], new_features[::-1]):
        layer = tf.keras.layers.Dense(new_feature, 'tanh')(layer)
        print(layer)

        if is_training:
            layer = tf.keras.layers.Dropout(drop_prob)(layer)

        layer = tf.keras.layers.LSTM(unit, return_sequences=True)(layer)
        print(layer)

        if is_training:
            layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.Dense(feature_length, 'tanh')(layer)
    print(layer)

    layer = tf.keras.layers.Flatten()(layer)
    print(layer)

    # exit()
    if is_training:
        return layer, drop_prob

    return layer


def create_conv_model(fingerprint_input, model_settings, is_training=False):
    time_length = model_settings['spectrogram_length']
    feature_length = model_settings['feature_length']

    if is_training:
        drop_prob = tf.compat.v1.placeholder(tf.float32, [])

    layer = tf.keras.layers.Reshape([time_length, feature_length, 1])(fingerprint_input)
    print(layer)

    layer = tf.keras.layers.Conv2D(40, [8, 12], 1, 'same', activation='relu')(layer)
    print(layer)

    if is_training:
        layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.AveragePooling2D([4, 4], 2, 'same')(layer)
    print(layer)

    layer = tf.keras.layers.Conv2D(80, [4, 6], 1, 'same', activation='relu')(layer)
    print(layer)

    if is_training:
        layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.AveragePooling2D([4, 4], 2, 'same')(layer)
    print(layer)

    layer = tf.keras.layers.Flatten()(layer)
    print(layer)

    layer = tf.keras.layers.Dense(80, name='encoded')(layer)
    print(layer)

    if is_training:
        layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.Dense(22 * 60 * 80)(layer)
    print(layer)

    if is_training:
        layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.Reshape([22, 60, 80])(layer)
    print(layer)

    layer = tf.keras.layers.UpSampling2D([2, 2])(layer)
    print(layer)

    layer = tf.keras.layers.Conv2DTranspose(40, [4, 6], 1, 'same', activation='relu')(layer)
    print(layer)

    if is_training:
        layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.UpSampling2D([2, 2])(layer)
    print(layer)

    layer = tf.keras.layers.Conv2DTranspose(1, [8, 12], 1, 'same')(layer)
    print(layer)

    layer = tf.keras.layers.Flatten()(layer)
    print(layer)

    #exit()
    if is_training:
        return layer, drop_prob

    return layer


def create_model(fingerprint_input, model_settings, is_training=False):
    time_length = model_settings['spectrogram_length']
    feature_length = model_settings['feature_length']

    if is_training:
        drop_prob = tf.compat.v1.placeholder(tf.float32, [])

    layer = tf.keras.layers.Reshape([time_length, feature_length])(fingerprint_input)
    units = [feature_length] + [320, 160, 80, 40]

    for unit in units[:-1]:
       layer = tf.keras.layers.Dense(unit, 'relu')(layer)
       print(layer)
       if is_training:
           layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.Dense(units[-1], 'relu', name='encoded')(layer)
    print(layer)
    if is_training:
        layer = tf.keras.layers.Dropout(drop_prob)(layer)

    for unit in units[1:-1][::-1]:
       layer = tf.keras.layers.Dense(unit, 'relu')(layer)
       print(layer)
       if is_training:
           layer = tf.keras.layers.Dropout(drop_prob)(layer)

    layer = tf.keras.layers.Dense(units[0], 'tanh', name='decoded')(layer)
    layer = tf.keras.layers.Flatten()(layer)
    print(layer)

    if is_training:
        return layer, drop_prob
    return layer
