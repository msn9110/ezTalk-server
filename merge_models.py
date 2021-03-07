import argparse
import os.path

import tensorflow as tf
from utils.tf_graph_util import convert_variables_to_constants

import models

FLAGS = None

tf = tf.compat.v1


def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            tf.logging.info(parameters_string)
        tf.logging.info(
            "Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


def create_inputs(model_settings):
    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    decoded_sample_data = tf.audio.decode_wav(
        wav_data_placeholder,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_data')
    import my_signal
    n_mfcc = model_settings['feature_length']
    fingerprint_input = my_signal.mfcc(decoded_sample_data.audio,
                                       model_settings['sample_rate'],
                                       frame_length=model_settings['window_size_samples'],
                                       frame_step=model_settings['window_stride_samples'],
                                       num_mfcc=n_mfcc)
    reshaped_input = tf.reshape(fingerprint_input, [
        -1, model_settings['fingerprint_size']
    ], name='fingerprint_input')

    return reshaped_input


def create_inference_graph(model_settings,
                           model_architecture):
    g = tf.Graph()
    with g.as_default():
        input_l = tf.placeholder(tf.float32, [1, model_settings['fingerprint_size']],
                                 name='input')
        logits = models.create_model(
            input_l, model_settings, model_architecture, is_training=False, )

    # Create an output to use for inference.
    return logits, g


def main():
    model_settings = models.prepare_model_settings(
        0, FLAGS.sample_rate, FLAGS.clip_duration_ms,
        FLAGS.window_size_ms, FLAGS.window_stride_ms)

    ckpts = list(FLAGS.ckpts.split(','))
    graph_defs = []
    modes = ['tops', 'mids', 'bots']
    if len(ckpts) == 2:
        modes = ['pref', 'suff']
    for i, label_count in enumerate(map(int, FLAGS.label_counts.split(','))):
        model_settings['label_count'] = label_count + 2
        # Create the model and load its weights.
        logits, g = \
            create_inference_graph(model_settings, FLAGS.model_architecture)
        with g.as_default():
            tf.nn.softmax(logits, name='softmax'.format(i + 1))
            with tf.Session(graph=g) as sess:
                models.load_variables_from_checkpoint(sess, ckpts[i])
                frozen_graph_def = convert_variables_to_constants(
                    sess, sess.graph_def, ['softmax'])
                graph_defs.append(frozen_graph_def)

    import cmodels.idata as cmi
    outputs = []
    input_l = create_inputs(model_settings)
    for i, m in enumerate(modes):
        [o] = tf.import_graph_def(graph_defs[i], input_map={'input': input_l},
                                  return_elements=['softmax:0'], name=m)

        target = o + cmi.epsilon
        outputs.append(target)

    phoneme = tf.concat(outputs, -1, name='phoneme')

    import cmodels.model as cmm

    input_units = list(map(len, cmi.indexes[1:4]))

    if len(ckpts) == 2:
        input_units = list(map(len, cmi.indexes[4:]))
    n_inputs = sum(input_units)
    model_settings = cmm.prepare_model_settings(n_inputs,
                                                len(cmi.zindexes),
                                                input_units)

    # load cs graph
    g = tf.Graph()
    with g.as_default():
        in_ = tf.placeholder(tf.float32, [1, model_settings['num_inputs']], name='cs_inputs')
        with tf.Session(graph=g) as sess:
            logits = cmm.create_model(in_, model_settings)
            cmm.load_variables_from_checkpoint(sess, FLAGS.cs)
            frozen_graph_def = convert_variables_to_constants(
                sess, sess.graph_def, [logits.name[:-2]])

    # back to default graph
    [output] = tf.import_graph_def(frozen_graph_def, input_map={'cs_inputs': phoneme},
                                   return_elements=[logits.name], name='')
    res = tf.nn.softmax(output, name='labels_softmax')
    print(res)

    with tf.Session() as sess:
        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = convert_variables_to_constants(
            sess, sess.graph_def, ['labels_softmax'])
        tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname(FLAGS.output_file),
            os.path.basename(FLAGS.output_file),
            as_text=False)
    tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=900,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=32.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long the stride is between spectrogram timeslices', )
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--label_counts',
        type=str,
        default='22,4,14',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--output_file', type=str, help='Where to save the frozen graph.',
        default='ckpts/cs.pb')
    parser.add_argument('--ckpts', type=str, default='ckpts/1/conv.ckpt-4401')
    parser.add_argument('--cs', type=str, default='ckpts/0/dnn.ckpt-4601')
    FLAGS, unparsed = parser.parse_known_args()
    main()
