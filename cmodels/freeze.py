import argparse
import os.path
import sys

import idata as input_data
import model as models

from cmodels import tf

FLAGS = None


def create_inference_graph():
    input_units = list(map(len, input_data.indexes[1:4]))

    if FLAGS.pre_suf:
        input_units = list(map(len, input_data.indexes[4:]))
    n_inputs = sum(input_units)
    model_settings = models.prepare_model_settings(n_inputs,
                                                   len(input_data.zindexes),
                                                   input_units)

    in_tensor = tf.placeholder(tf.float32, [None, model_settings['num_inputs']],
                               name='input')
    logits = models.create_model(in_tensor, model_settings, is_training=False)

    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')


def main(_):
    # Create the model and load its weights.
    sess = tf.InteractiveSession()
    create_inference_graph()
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
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
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--output_file', type=str, help='Where to save the frozen graph.')
    parser.add_argument(
        '-ps', '--pre_suf',
        action='store_true',
        help='use pre suf')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
