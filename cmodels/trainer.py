import argparse
import os.path
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin

import idata as input_data
import model as models
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
gfile = tf.io.gfile
FLAGS = None


def main(_):
    # ---------------early stopping---------------
    max_val = 0.0
    max_val_ = 0.0
    cp_path = ''
    current_patient = 1
    my_patient = 4

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
    # ------------------------------------------------
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=None)
    input_units = list(map(len, input_data.indexes[1:4]))

    if FLAGS.pre_suf:
        input_units = list(map(len, input_data.indexes[4:]))
    n_inputs = sum(input_units)
    model_settings = models.prepare_model_settings(n_inputs,
                                                   len(input_data.zindexes),
                                                   input_units)
    data_processor = input_data.DataProcessor(FLAGS.data_path,
                                              FLAGS.validation_percentage,
                                              FLAGS.testing_percentage,
                                              FLAGS.test_set)

    fingerprint_size = model_settings['num_inputs']
    label_count = model_settings['num_outputs']

    # 100 * 43
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('loss'):
        cross_entropy = tf.keras.losses.categorical_crossentropy(
            ground_truth_input, logits, from_logits=True)
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
        # new
        weights = tf.math.sqrt(tf.cast(weights, tf.float32)) + 1.0
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
        step_rate = 800
        decay = 0.95
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)
        train_step = optimizer.minimize(cross_entropy_mean)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=my_patient + 1)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
    testing_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/testing')

    tf.global_variables_initializer().run()

    FLAGS.model_architecture = 'dnn'
    start_step = 1
    checkpoint_path = os.path.join(FLAGS.train_dir, 'ckpt')
    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ' % start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir, '{0:s}.pbtxt'.format(FLAGS.model_architecture))

    # Save list of words.
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, 'labels.txt'),
            'w') as f:
        f.write('\n'.join(input_data.valid_zhuyins))

    final = 0
    epoch = 0
    # Training loop.
    training_steps_max = FLAGS.how_many_training_steps
    for training_step in xrange(start_step, training_steps_max + 1):
        # Pull the audio samples we'll use for training.
        _, train_fingerprints, train_ground_truth = data_processor.get_data(
            FLAGS.batch_size, 0, 'training')
        # Run the graph with this batch of training data.
        train_summary, train_accuracy, loss_value, index_value, *_ = sess.run(
            [
                merged_summaries, evaluation_step, loss, index_loss_mean, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                dropout_prob: 0.5
            })

        learning_rate_value = sess.run(optimizer._lr)

        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, loss %f, avg. index %.1f val %.1f%%, %d' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         loss_value, index_value, max_val_, current_patient))
        is_last_step = (training_step == training_steps_max)
        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            epoch += 1
            train_writer.add_summary(train_summary, epoch)

            set_size = data_processor.set_size('validation')
            total_accuracies = [0] * len(top_ks)
            total_accuracy = 0
            tce = 0.0
            tloss = 0.0
            total_ranking = 0.0
            total_conf_matrix = None
            for i in xrange(0, set_size, FLAGS.batch_size):
                _, validation_fingerprints, validation_ground_truth =\
                    data_processor.get_data(
                    FLAGS.batch_size, i, 'validation')
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                ranking, vce, vloss, conf_matrix, validation_accuracy, \
                *validation_accuracies = sess.run(
                    [index_loss_mean, cross_entropy_mean, loss,
                     confusion_matrix, evaluation_step] + top_ks,
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
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

            set_size = data_processor.set_size('testing')
            if set_size:
                total_accuracy = 0
                tce = 0.0
                tloss = 0.0
                total_ranking = 0.0
                for i in xrange(0, set_size, FLAGS.batch_size):
                    _, testing_fingerprints, testing_ground_truth = (
                        data_processor.get_data(FLAGS.batch_size, i, 'testing'))

                    ranking, testce, testloss, testing_accuracy = sess.run(
                        [index_loss_mean, cross_entropy_mean, loss,
                         evaluation_step],
                        feed_dict={
                            fingerprint_input: testing_fingerprints,
                            ground_truth_input: testing_ground_truth,
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

    set_size = data_processor.set_size('testing')
    if set_size > 0:

        tf.logging.info('set_size=%d' % set_size)
        total_accuracies = [0] * len(top_ks)
        total_accuracy = 0
        total_index = 0
        correct_files = []
        total_conf_matrix = None
        for i in xrange(0, set_size, FLAGS.batch_size):
            files, test_fingerprints, test_ground_truth = data_processor.get_data(
                FLAGS.batch_size, i, 'testing')
            conf_matrix,  corrections, index_value, test_accuracy, *test_accuracies = sess.run(
                [confusion_matrix, correct_prediction, index_loss_mean, evaluation_step] + top_ks,
                feed_dict={
                    fingerprint_input: test_fingerprints,
                    ground_truth_input: test_ground_truth,
                    dropout_prob: 0.0  # keep all
                })
            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size
            total_index += (index_value * batch_size) / set_size
            correct_files += [f for f, v in zip(files, corrections) if v]
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
        tf.logging.info('Final test accuracy = %.1f%% %.1f (N=%d)' % (total_accuracy * 100,
                                                                      total_index,
                                                                  set_size))
        """
        with open('correct_tests.txt', 'w') as f:
            f.write('\n'.join(correct_files))
        tensor = sess.graph.get_tensor_by_name('scalar:0')
        tf.logging.info(sess.run(tensor),)
        tensor = sess.graph.get_tensor_by_name('bias:0')
        tf.logging.info(sess.run(tensor))
        tf.logging.info(final)
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p', '--data_path',
        type=str,
        default='/home/dmcl/dataset/train/_0/train_data.json',
        help='train json path')
    parser.add_argument(
        '-ts', '--test_set',
        type=str,
        default='',
        help='json path of custom testing set')
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
        '--how_many_training_steps',
        type=int,
        default=50000,
        help='How many training loops to run', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=200,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training at start.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10000,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/home/dmcl/system/tools/.temp/logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/home/dmcl/system/tools/.temp/cmds',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '-st', '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pre-trained model before any training.')
    parser.add_argument(
        '-s', '--suffix',
        type=str,
        default='',
        help='suffix')
    parser.add_argument(
        '-chk', '--check_nans',
        action='store_true',
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '-ps', '--pre_suf',
        action='store_true',
        help='use pre suf')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.summaries_dir += FLAGS.suffix
    FLAGS.train_dir += FLAGS.suffix
    if not FLAGS.data_path:
        raise ValueError('Please Input data path')
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
