from tensorflow.python.keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Reshape, Dropout, LSTM, ReLU, BatchNormalization, Bidirectional
import tensorflow as tf


tf = tf.compat.v1


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms):
    """Calculates common settings needed for all models.

    Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.

    Returns:
    Dictionary containing common settings.
    """
    from my_signal import feature_length
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = feature_length * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'feature_length': feature_length,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training):
    if model_architecture == 'conv':
        return create_conv_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'lstm':
        return create_lstm_model(fingerprint_input, model_settings, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "single_fc", "conv"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_lstm_model(fingerprint_input, model_settings, is_training):

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    # information for constructing model
    input_frequency_size = model_settings['feature_length']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    b_norm_axis = 1

    print(fingerprint_input)
    
    x = Reshape([input_time_size, input_frequency_size])(fingerprint_input)
    print(x)

    '''x = BatchNormalization(b_norm_axis)(x)
    print(x)'''

    '''
    x = Conv2D(128, [5, 4], [1, 1], 'same')(x)
    print(x)
    x = tf.reduce_sum(x, -1)
    print(x)
    x = BatchNormalization(-1)(x)
    '''

    x = LSTM(80, return_sequences=True, name='LSTM_1')(x)
    print(x)
    
    # reduce feature
    n_feature = 40

    if is_training:
        x = Dropout(dropout_prob)(x)

    '''x = BatchNormalization(b_norm_axis)(x)'''
    x = LSTM(n_feature, return_sequences=True, name='LSTM_2')(x)
    print(x)

    if is_training:
        x = Dropout(dropout_prob)(x)

    # reduce feature
    n_feature = 20

    '''x = BatchNormalization(b_norm_axis)(x)'''
    x = LSTM(n_feature, return_sequences=True, name='LSTM_3')(x)
    print(x)

    '''x = BatchNormalization(b_norm_axis)(x)'''
    s = Dense(1, 'sigmoid', name='weights')(x)
    v = Dense(label_count,)(x)

    scalars = s
    print(scalars)

    x = v * scalars
    print(x)

    # voting
    logits = tf.reduce_sum(x, 1, name='voting')
    '''logits = tf.nn.elu(logits)'''
    print(logits)

    if is_training:
        return logits, dropout_prob
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training, shortcut=False):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['feature_length']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    filters = 128

    # build model
    fingerprint_4d = Reshape([input_time_size, input_frequency_size, 1])(fingerprint_input)
    activation = ReLU()

    # 78 * 40 - 64
    first_conv = Conv2D(filters, [20, 8], 1, 'same', use_bias=True)(fingerprint_4d)
    print(first_conv)
    first_relu1 = activation(first_conv)

    if is_training:
        first_dropout = Dropout(dropout_prob)(first_relu1)
    else:
        first_dropout = first_relu1
    print(first_dropout)
    # 39 * 20 - 64
    max_pool = MaxPool2D([3, 3], 2, 'same')(first_dropout)
    print(max_pool)
    filters *= 2

    # 39 * 20 - 64
    second_conv = Conv2D(filters, [10, 4], 1, 'same', use_bias=True)(max_pool)
    print(second_conv)
    second_relu1 = activation(second_conv)
    # shortcut connection
    first_concat = tf.concat([second_relu1, max_pool], 3)

    second_conv2 = Conv2D(filters, [10, 4], 1, 'same', use_bias=True)(first_concat)
    second_relu2 = activation(second_conv2) if shortcut else second_relu1
    if is_training:
        second_dropout = Dropout(dropout_prob)(second_relu2)
    else:
        second_dropout = second_relu2
    print(second_dropout)

    max_pool_2 = MaxPool2D([3, 3], 2, 'same')(second_dropout)
    print(max_pool_2)
    filters *= 2

    third_conv1 = Conv2D(128, [5, 2], 1, 'same', use_bias=True)(max_pool_2)
    print(third_conv1)
    third_relu1 = activation(third_conv1)

    x_flatten = Flatten()(third_relu1)
    final_fc = Dense(label_count)(x_flatten)
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc
