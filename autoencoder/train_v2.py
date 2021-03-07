import argparse
import os.path
import sys, os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import autoencoder
import input_data2 as input_data
import models
from tensorflow.python.platform import gfile


tf = tf.compat.v1
FLAGS = None
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main(_):
    min_val = -1.0
    min_val_ = -1.0
    cp_path = ''
    current_patient = 1
    my_patient = 3

    def early_stopping(val, cp, patient=3):
        nonlocal min_val, min_val_, cp_path, current_patient
        if min_val == -1.0:
            min_val = val
            min_val_ = round(val, 1)
            cp_path = cp
            return False

        if val < min_val_:
            min_val_ = val
            cp_path = cp

        val = round(val, 1)
        if val < min_val:
            min_val = val
            current_patient = 1
        else:
            current_patient += 1
        return True if current_patient >= patient else False

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=None)

    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        1,
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms)
    audio_processor = input_data.AudioProcessor(
        FLAGS.mode, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage, FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings, FLAGS.test_dir)
    fingerprint_size = model_settings['fingerprint_size']
    model_settings['label_count'] = len(audio_processor.words_list)
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    training_set_settings = {
        'how_many': FLAGS.batch_size,
        'offset': 0,
        'background_frequency': FLAGS.background_frequency,
        'background_volume_range': FLAGS.background_volume,
        'time_shift': time_shift_samples,
        'mode': 'training'
    }
    audio_processor.start_generator(training_set_settings,)

    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    tf.logging.info(training_steps_list)

    # 100 * 3920
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, dropout_prob = autoencoder.create_blstm_model(
        fingerprint_input,
        model_settings,
        is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='groundtruth_input')

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('loss'):
        my_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(ground_truth_input, logits), -1)
        loss = tf.reduce_mean(my_loss)
    tf.summary.scalar('my_loss', loss)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    #checks = tf.add_check_numerics_ops()
    control_dependencies = []
    if not FLAGS.check_nans:
        control_dependencies = []

    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):

        lr = FLAGS.learning_rate
        step_rate = 400
        decay = 0.9
        learning_rate = tf.train.exponential_decay(lr, global_step - 1, step_rate, decay,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(my_loss)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=my_patient + 1)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
    testing_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/testing')

    init = tf.global_variables_initializer()
    sess.run(init)

    start_step = 1
    checkpoint_path = os.path.join(FLAGS.train_dir,
                                   FLAGS.model_architecture)
    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = sess.run(global_step)

    tf.logging.info('Training from step: %d ' % start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    # Save list of words.
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, 'labels.txt'),
            'w') as f:
        f.write('\n'.join(audio_processor.words_list))

    final = 0
    epoch = 0
    # Training loop.
    training_steps_max = sum(training_steps_list)
    for training_step in xrange(start_step, training_steps_max + 1):

        # Pull the audio samples we'll use for training.
        train_fingerprints, train_ground_truth = audio_processor.get_data_nb(
            mode='training'
        )
        # Run the graph with this batch of training data.
        train_summary, loss_value, *_ = sess.run(
            [
                merged_summaries, loss, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                dropout_prob: 0.3
            })

        learning_rate_value = sess.run(optimizer._lr)

        tf.logging.info('Step #%d: rate %f, loss %f, %d' %
                        (training_step, learning_rate_value,
                         loss_value, current_patient))
        is_last_step = (training_step == training_steps_max)
        if (training_step % 100 == 0) or is_last_step:

            pass

        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            epoch += 1
            train_writer.add_summary(train_summary, epoch)

            set_size = audio_processor.set_size('validation')
            tloss = 0.0
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth = (
                    audio_processor.get_data_nb(how_many=FLAGS.batch_size, offset=i,
                                                mode='validation'))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                vloss, *_ = sess.run(
                    [loss],
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        dropout_prob: 0.0  # keep all
                    })
                batch_size = min(FLAGS.batch_size, set_size - i)
                tloss += (vloss * batch_size) / set_size
            val = tloss

            # validation summary
            # Create a new Summary object with your measure
            summary = tf.Summary()
            summary.value.add(tag="my_loss", simple_value=tloss)
            validation_writer.add_summary(summary, epoch)

            # Save the model checkpoint periodically.
            tf.logging.info('Saving to "%s-%d"' % (checkpoint_path + '.ckpt', training_step))
            cp_path_ = '%s-%d' % (checkpoint_path + '.ckpt', training_step)
            saver.save(sess, checkpoint_path + '.ckpt', global_step=training_step)
            final = training_step + 1

            set_size = audio_processor.set_size('testing') #* 0
            if set_size:
                tloss = 0.0
                for i in xrange(0, set_size, FLAGS.batch_size):
                    testing_fingerprints, testing_ground_truth = (
                        audio_processor.get_data_nb(how_many=FLAGS.batch_size, offset=i,
                                                    mode='testing'))

                    testloss, *_ = sess.run(
                        [loss],
                        feed_dict={
                            fingerprint_input: testing_fingerprints,
                            ground_truth_input: testing_ground_truth,
                            dropout_prob: 0.0  # keep all
                        })
                    batch_size = min(FLAGS.batch_size, set_size - i)
                    tloss += (testloss * batch_size) / set_size

                # testing summary
                # Create a new Summary object with your measure
                summary = tf.Summary()
                summary.value.add(tag="my_loss", simple_value=tloss)
                testing_writer.add_summary(summary, epoch)

            if early_stopping(val, cp_path_, my_patient):
                tf.logging.info('Current patient %d, %.1f' % (current_patient, round(val, 1)))
                tf.logging.info('%s : %.1f' % (cp_path, min_val))
                break
            tf.logging.info('Current patient %d, %.1f' % (current_patient, round(val, 1)))
            tf.logging.info('%s : %.1f' % (cp_path, min_val))
    # restore best weights and save
    models.load_variables_from_checkpoint(sess, cp_path)

    saver.save(sess, checkpoint_path + '.ckpt', global_step=final)

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        help='0,1,2,3,4')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/hungshing/FastData/ezTalk/users/msn9110/voice_data/training/_0',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/home/hungshing/FastData/ezTalk/users/msn9110/voice_data/testing',
        help="""\
          Custom testing set
          """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
      How loud the background noise should be, between 0 and 1.
      """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in.
      """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=0,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
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
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='15000,3000',
        help='How many training loops to run', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results. and Save model checkpoint every'
             ' save_steps.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training at start.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='.temp/logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='.temp/ckpts',
        help='Directory to write event logs and checkpoint.')
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
        '-chk', '--check_nans',
        action='store_true',
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)

