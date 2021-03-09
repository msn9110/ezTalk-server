import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Reshape, ReLU, ELU


def prepare_model_settings(num_inputs, num_outputs, input_units):
    """
    :param num_inputs:
    :param num_outputs:
    :return: settings with type dict
    """
    return {
        'input_units': input_units,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs
    }


def create_model(fingerprint_input, model_settings, is_training=False):
    input_units = model_settings['input_units']
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    num_hidden_nodes = []
    regulizer = tf.keras.regularizers.l1_l2(l1=0., l2=0.)

    input_l = Reshape([model_settings['num_inputs']], name='input')(fingerprint_input)

    print(input_l)
    # tf.random.normal([])
    # 1 / ln 2 ~ 1.442695
    scalars = tf.Variable(tf.random.normal([]),
                          dtype=tf.float32, name='scalar', trainable=True)
    bias = tf.Variable(initial_value=0.05, dtype=tf.float32, name='bias',
                       trainable=True)
    processed_tensor = tf.add(tf.multiply(scalars, tf.math.log(input_l)), bias, 'processed')
    print(processed_tensor)

    prev_l = processed_tensor

    relu_count = 1
    dense_count = 1

    # construct full-connected hidden layers
    for units in num_hidden_nodes:
        if units == 0:
            prev_l = ReLU(name='relu{0:d}'.format(relu_count))(prev_l)
            relu_count += 1
            print(prev_l)
            continue
        elif units < 0:
            prev_l = ELU(name='elu{0:d}'.format(relu_count))(prev_l)
            relu_count += 1
            print(prev_l)
            continue

        hidden = Dense(units, kernel_regularizer=regulizer,
                       name='dense{0:d}'.format(dense_count))(prev_l)
        dense_count += 1
        print(hidden)
        if is_training:
            dropout_l = Dropout(dropout_prob)(hidden)
        else:
            dropout_l = hidden
        prev_l = dropout_l

    output_l = Dense(model_settings['num_outputs'], name='output')(prev_l)
    if is_training:
        return output_l, dropout_prob
    else:
        return output_l


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)
