import argparse
import os.path
import sys

from utils.tf_graph_util import convert_variables_to_constants
import models

from _tf_ import tf


FLAGS = None


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           model_architecture):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    """

    words_list = wanted_words.split(',')
    print(len(words_list), 'labels')
    model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms)

    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    decoded_sample_data = tf.audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
    import my_signal
    feature_length = model_settings['feature_length']
    fingerprint_input = my_signal.mfcc(decoded_sample_data.audio,
                                model_settings['sample_rate'],
                                frame_length=model_settings['window_size_samples'],
                                frame_step=model_settings['window_stride_samples'],
                                num_mfcc=feature_length)
    fingerprint_frequency_size = feature_length
    fingerprint_time_size = model_settings['spectrogram_length']
    reshaped_input = tf.reshape(fingerprint_input, [
      -1, fingerprint_time_size * fingerprint_frequency_size
    ])

    logits = models.create_model(
      reshaped_input, model_settings, model_architecture, is_training=False,)

    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')


def main(_):

    # Create the model and load its weights.
    sess = tf.InteractiveSession()
    create_inference_graph(FLAGS.wanted_words, FLAGS.sample_rate,
                         FLAGS.clip_duration_ms,
                         FLAGS.window_size_ms, FLAGS.window_stride_ms,
                         FLAGS.model_architecture)
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    #print('\n'.join([n.name for n in sess.graph_def.node]))

    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = convert_variables_to_constants(
      sess, sess.graph_def, ['labels_softmax'])
    tf.io.write_graph(
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
      help='Expected sample rate of the wavs',)
    parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
      '--window_size_ms',
      type=float,
      default=32.0,
      help='How long each spectrogram timeslice is',)
    parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices',)
    parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
    parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
      '--output_file', type=str, help='Where to save the frozen graph.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
