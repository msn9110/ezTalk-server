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
    begin = 0
    input_tensors = []

    for name_num, unit in enumerate(input_units, 1):
        input_tensor = tf.slice(input_l, [0, begin], [-1, unit], name='input_{0:d}'.format(name_num))
        print(input_tensor)
        input_tensors.append(input_tensor)
        begin += unit

    processed_tensors = []

    for name_num, tensor in enumerate(input_tensors, 1):
        # tf.random.normal([])
        # 1 / ln 2 ~ 1.442695
        scalars = tf.Variable(tf.random.normal([]),
                              dtype=tf.float32, name='scalar_{0:d}'.format(name_num), trainable=True)
        bias = tf.Variable(initial_value=0.05, dtype=tf.float32, name='bias_{0:d}'.format(name_num),
                           trainable=True)
        t = tf.add(tf.multiply(scalars, tf.math.log(tensor)), bias, 'processed_{0:d}'.format(name_num))
        print(t)
        processed_tensors.append(t)

    hidden_units_l = [
        [],
        [],
        []
    ]

    hidden_tensors = []
    for name_num, tensor in enumerate(processed_tensors):
        hidden_units = hidden_units_l[name_num]
        name_num += 1
        for units in hidden_units:
            if units > 0:
                tensor = tf.keras.layers.Dense(units)(tensor)
                if is_training:
                    tensor = tf.keras.layers.Dropout(dropout_prob)(tensor)
                else:
                    if units == 0:
                        tensor = tf.keras.layers.ReLU()(tensor)
                    else:
                        tensor = tf.keras.layers.ELU()(tensor)
        hidden_tensors.append(tensor)

    if not hidden_tensors:
        hidden_tensors = processed_tensors

    prev_l = tf.concat(hidden_tensors, -1)
    print(prev_l)

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
