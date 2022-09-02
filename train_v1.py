import argparse
import os.path
import sys, os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import input_data
import models
gfile = tf.io.gfile

from config import write_log, set_pid
FLAGS = None
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(_):
    max_val = 0.0
    max_val_ = 0.0
    cp_path = ''
    current_patient = 1
    my_patient = 3

    def early_stopping(val_acc, cp, patient=3):
        nonlocal max_val, max_val_, cp_path, current_patient
        if val_acc > max_val_:
            max_val_ = val_acc
            cp_path = cp
        if max_val_ - val_acc >= 2.5:
            return True

        val_acc = round(val_acc)
        if val_acc > max_val:
            max_val = val_acc
            current_patient = 1
        else:
            current_patient += 1
        return True if current_patient >= patient else False

    set_pid('worker')
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

    logits, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')
    sim_input = tf.placeholder(
        tf.float32, [None, label_count], name='sim_input')

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('loss'):
        cross_entropy = tf.keras.losses.categorical_crossentropy(
            sim_input, logits, from_logits=True)
        print(cross_entropy)
        predicted_indexes = tf.cast(
            tf.argsort(
                tf.argsort(logits, direction='DESCENDING', stable=True)),
            tf.float32)
        index_loss = ground_truth_input * predicted_indexes
        # 0 ~ 395
        weights = tf.reduce_sum(index_loss, 1)

        index_loss_mean = tf.reduce_mean(
            weights
        )
        weights = tf.math.sqrt(tf.cast(weights, tf.float32))
        #weights = tf.minimum(weights, 1.5)
        my_loss = tf.reduce_mean(
            tf.pow(weights, 1.0) *
            cross_entropy
        )
        regulized_losses = tf.reduce_mean(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = my_loss  # cross_entropy_mean + regulized_losses
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.summary.scalar('ranking', index_loss_mean)
    tf.summary.scalar('my_loss', loss)

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices,
                                           num_classes=label_count)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # top1 - top9
    top_ks = []
    for k in range(1, 11, 2):
        in_topk = tf.nn.in_top_k(logits, expected_indices, k=k)
        topk = tf.reduce_mean(tf.cast(in_topk, tf.float32))
        top_ks.append(topk)
    tf.summary.scalar('accuracy', evaluation_step)

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
        train_step = optimizer.minimize(cross_entropy_mean)

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
                                   'ckpt')
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
        train_fingerprints, train_ground_truth, sim_vecs = audio_processor.get_data_nb(
            mode='training'
        )
        # Run the graph with this batch of training data.
        train_summary, train_accuracy, loss_value, index_value, *_ = sess.run(
            [
                merged_summaries, evaluation_step, loss, index_loss_mean, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                sim_input: sim_vecs,
                dropout_prob: 0.75
            })

        learning_rate_value = sess.run(optimizer._lr)

        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, loss %f, avg. index %.1f val %.1f%%, %d' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         loss_value, index_value, max_val_, current_patient))
        is_last_step = (training_step == training_steps_max)
        if (training_step % 100 == 0) or is_last_step:

            write_log('step', str(training_step))

        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            epoch += 1
            train_writer.add_summary(train_summary, epoch)

            set_size = audio_processor.set_size('validation')
            total_accuracies = [0] * len(top_ks)
            total_accuracy = 0
            tce = 0.0
            tloss = 0.0
            total_ranking = 0.0
            total_conf_matrix = None
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth, sim_vecs = (
                    audio_processor.get_data_nb(how_many=FLAGS.batch_size, offset=i,
                                                mode='validation'))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                ranking, vce, vloss, conf_matrix, validation_accuracy, \
                *validation_accuracies = sess.run(
                    [index_loss_mean, cross_entropy_mean, loss,
                     confusion_matrix, evaluation_step] + top_ks,
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        sim_input: sim_vecs,
                        dropout_prob: 0.0  # keep all
                    })
                batch_size = min(FLAGS.batch_size, set_size - i)
                tce += (vce * batch_size) / set_size
                tloss += (vloss * batch_size) / set_size
                total_ranking += (ranking * batch_size) / set_size
                total_accuracy += (validation_accuracy * batch_size) / set_size
                for j in xrange(len(total_accuracies)):
                    total_accuracies[j] += (validation_accuracies[j] * batch_size) / set_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix

            # validation summary
            # Create a new Summary object with your measure
            summary = tf.Summary()
            summary.value.add(tag="cross_entropy", simple_value=tce)
            summary.value.add(tag="my_loss", simple_value=tloss)
            summary.value.add(tag="ranking", simple_value=total_ranking)
            summary.value.add(tag="accuracy", simple_value=total_accuracy)
            validation_writer.add_summary(summary, epoch)

            val_accs = [_ * 100 for _ in total_accuracies]
            tf.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
            for j, val_acc in enumerate(val_accs):
                tf.logging.info('Step %d: Validation top %d = %.1f%% (N=%d)' %
                                (training_step, 2 * j + 1, val_acc, set_size))
            val_acc = total_accuracy * 100
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, val_acc, set_size))

            # Save the model checkpoint periodically.
            tf.logging.info('Saving to "%s-%d"' % (checkpoint_path, training_step))
            cp_path_ = '%s-%d' % (checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)
            final = training_step + 1

            set_size = audio_processor.set_size('testing') #* 0
            if set_size:
                total_accuracy = 0
                tce = 0.0
                tloss = 0.0
                total_ranking = 0.0
                for i in xrange(0, set_size, FLAGS.batch_size):
                    testing_fingerprints, testing_ground_truth, sim_vecs = (
                        audio_processor.get_data_nb(how_many=FLAGS.batch_size, offset=i,
                                                    mode='testing'))

                    ranking, testce, testloss, testing_accuracy = sess.run(
                        [index_loss_mean, cross_entropy_mean, loss,
                         evaluation_step],
                        feed_dict={
                            fingerprint_input: testing_fingerprints,
                            ground_truth_input: testing_ground_truth,
                            sim_input: sim_vecs,
                            dropout_prob: 0.0  # keep all
                        })
                    batch_size = min(FLAGS.batch_size, set_size - i)
                    tce += (testce * batch_size) / set_size
                    tloss += (testloss * batch_size) / set_size
                    total_ranking += (ranking * batch_size) / set_size
                    total_accuracy += (testing_accuracy * batch_size) / set_size

                # testing summary
                # Create a new Summary object with your measure
                summary = tf.Summary()
                summary.value.add(tag="cross_entropy", simple_value=tce)
                summary.value.add(tag="my_loss", simple_value=tloss)
                summary.value.add(tag="ranking", simple_value=total_ranking)
                summary.value.add(tag="accuracy", simple_value=total_accuracy)
                testing_writer.add_summary(summary, epoch)

            if early_stopping(val_acc, cp_path_, my_patient):
                tf.logging.info('Current patient %d, %.1f%%' % (current_patient, round(val_acc)))
                tf.logging.info('%s : %.1f%%' % (cp_path, max_val_))
                break
            tf.logging.info('Current patient %d, %.1f%%' % (current_patient, round(val_acc)))
            tf.logging.info('%s : %.1f%%' % (cp_path, max_val_))
    # restore best weights and save
    models.load_variables_from_checkpoint(sess, cp_path)

    saver.save(sess, checkpoint_path, global_step=final)

    set_size = audio_processor.set_size('testing')
    if set_size > 0:

        tf.logging.info('set_size=%d' % set_size)
        total_accuracies = [0] * len(top_ks)
        total_accuracy = 0
        total_conf_matrix = None
        for i in xrange(0, set_size, FLAGS.batch_size):
            test_fingerprints, test_ground_truth, sim_vecs = audio_processor.get_data_nb(
                how_many=FLAGS.batch_size, offset=i, mode='testing')
            conf_matrix, test_accuracy, *test_accuracies = sess.run(
                [confusion_matrix, evaluation_step] + top_ks,
                feed_dict={
                    fingerprint_input: test_fingerprints,
                    ground_truth_input: test_ground_truth,
                    sim_input: sim_vecs,
                    dropout_prob: 0.0
                })
            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size
            for j in xrange(len(total_accuracies)):
                total_accuracies[j] += (test_accuracies[j] * batch_size) / set_size
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        tf.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
        for j in xrange(len(total_accuracies)):
            tf.logging.info('Final test top %d = %.1f%% (N=%d)' % (2 * j + 1,
                                                                   total_accuracies[j] * 100,
                                                                   set_size))
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                                 set_size))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', '-m',
        type=int,
        default=0,
        help='0,1,2,3,4')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--test_dir',
        type=str,
        default='',
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
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/speech_commands_train',
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

