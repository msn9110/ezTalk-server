import hashlib
import math
import os.path
import random
import re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from _tf_ import tf

gfile = tf.io.gfile

from phoneme import zelements, optionals, zindexes, idx_mode, sim_tables, sels


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def duplicate_data(origin_data, expand_mode=1):
    """
    Utility for over-sampling data
    :param origin_data: [{'label': 'a', 'file': path}, ...]
    :param expand_mode: 0 to expand as median size, otherwise as max size
    :return: over-sampled origin data
    """
    classified_data = {}
    for wav in origin_data:
        label, path = wav['label'], wav['file']
        if label in classified_data.keys():
            classified_data[label].append(path)
        else:
            classified_data[label] = [path]
    nums = [(k, len(classified_data[k]))
            for k in classified_data.keys()]
    nums = list(sorted(nums, key=lambda a: a[1]))
    if expand_mode:
        # max expand
        idx = len(nums) - 1
    else:
        # median expand
        idx = len(nums) // 2 if len(nums) % 2 \
            else (len(nums) - 1) // 2
    size = nums[idx][1]
    # duplicate data
    for i in range(idx):
        label, origin_size = nums[i]
        expand_value = size - origin_size
        expand_data = [classified_data[label][np.random.randint(0, origin_size)]
                       for _ in range(expand_value)]
        classified_data[label].extend(expand_data)
    new_data = [{'label': k, 'file': f}
                for k in classified_data.keys()
                for f in classified_data[k]]
    return new_data


def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.

    Args:
      wanted_words: List of strings containing the custom words.

    Returns:
      List with the standard silence and unknown tokens added.
    """
    return [UNKNOWN_WORD_LABEL, SILENCE_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage, salt=0):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = (((int(hash_name_hashed, 16) + salt) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
      filename: Path to the .wav file to load.

    Returns:
      Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = tf.io.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio sample data to a .wav audio file.

    Args:
      filename: Path to save the file to.
      wav_data: 2D array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        sample_rate_placeholder = tf.placeholder(tf.int32, [])
        wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
        wav_encoder = tf.audio.encode_wav(wav_data_placeholder,
                                               sample_rate_placeholder)
        wav_saver = tf.io.write_file(wav_filename_placeholder, wav_encoder)
        sess.run(
            wav_saver,
            feed_dict={
                wav_filename_placeholder: filename,
                sample_rate_placeholder: sample_rate,
                wav_data_placeholder: np.reshape(wav_data, (-1, 1))
            })


class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""
    from config import get_settings
    *_, settings = get_settings()
    training_default_values = settings['training_default_values']

    from multiprocessing import Manager
    BALANCE_DATA = training_default_values['over_sampling']
    results = Manager().dict({'validation': [], 'testing': []})
    generators = {'training': None, 'validation': None, 'testing': None}

    def __init__(self, mode, data_dir, silence_percentage, unknown_percentage,
                 validation_percentage, testing_percentage,
                 model_settings, test_dir=None, sim_mode=True):
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.mode = mode
        self.sim_mode = sim_mode and mode in [0, 4, 5]

        self.prepare_data_index(silence_percentage, unknown_percentage,
                                validation_percentage, testing_percentage)
        self.prepare_background_data()
        self.g = None
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.config = config
        self.prepare_processing_graph(model_settings)

        self.model_settings = model_settings
        self.training_set_settings = None
        self.queue = None
        self.sess = None

    def start_generator(self, training_set_settings=None, start=False):

        defaults = {
            'how_many': 100,
            'offset': 0,
            'background_frequency': 0.5,
            'background_volume_range': 0.1,
            'time_shift': 100,
            'mode': 'training'
        }

        training_set_settings = training_set_settings if training_set_settings \
            else defaults
        self.training_set_settings = training_set_settings

        import multiprocessing as mp
        from multiprocessing import Process, Queue

        q = Queue(maxsize=10)
        self.queue = q

        if not start:
            return

        mp.set_start_method('spawn', force=True)
        for mode in ['training', 'validation', 'testing']:

            p = Process(target=self._background_generator,
                        args=(mode,),
                        name='{} data generator'.format(mode))
            p.daemon = True
            self.generators[mode] = p
            p.start()

    def _background_generator(self, mode):

        print(mode, self.g, self.config)

        if mode in ['validation', 'testing']:
            sess = tf.Session(graph=self.g, config=self.config)
            print(mode, 'generator\'s session has been initialized successfully.')
            set_size = self.set_size(mode)
            batch_size = self.training_set_settings['how_many']
            results = []
            for i in xrange(0, set_size, batch_size):
                results.append(self.get_data(batch_size, i, mode=mode, sess=sess))
            sess.close()
            self.results[mode] = results
            tf.logging.info('background {0:s} generator generates {1:d}'.format(mode,
                                                                                len(self.results[mode])) )

        if mode == 'training':
            sess = tf.Session(graph=self.g, config=self.config)
            q = self.queue
            max_qsize = 10
            while True:
                if q.qsize() < max_qsize:
                    results = self.get_data(**self.training_set_settings, sess=sess)
                    q.put(results)
                else:
                    #todo wait
                    pass


    def get_data_nb(self, mode, how_many=100, offset=0,):

        if self.sess is None:
            sess = tf.Session(graph=self.g, config=self.config)
            self.sess = sess

        timeout_base = 0.0
        timeout = int(timeout_base * self.set_size(mode))
        if mode in ['validation', 'testing']:
            worker = self.generators[mode]
            if worker:
                worker.join(timeout)
                worker = None
                self.generators[mode] = worker

            try:
                res = self.results[mode][int(offset // how_many)]
            except IndexError:
                res = self.get_data(how_many, offset, mode=mode, sess=self.sess)
                self.results[mode] += [res]

            return res

        elif mode == 'training':
            timeout = 0
            q = self.queue
            try:
                results = q.get(timeout=timeout)
            except Exception:
                results = None

            if not results:
                args = dict(self.training_set_settings)
                results = self.get_data(**args, sess=self.sess)

            min_qsize = 5
            if q.qsize() < min_qsize:
                # todo notify
                pass
            return results

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           validation_percentage, testing_percentage):
        """Prepares a list of the samples organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.

        Args:
          silence_percentage: How much of the resulting data should be background.
          unknown_percentage: How much should be audio outside the wanted classes.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.

        Raises:
          Exception: If expected files are not found.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.
        wanted_words = set()
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}

        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case,
            #  since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue

            set_index = which_set(wav_path, validation_percentage,
                                  testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in zindexes:
                label = ([word] + zelements[word])[self.mode % 6]
                if self.mode == 6 and label not in sels:
                    continue
                wanted_words.add(label)
                self.data_index[set_index].append({'label': label, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': optionals[self.mode][UNKNOWN_WORD_INDEX],
                                                 'file': wav_path})

        if not wanted_words:
            raise Exception('No .wavs found at ' + search_path)

        labels = optionals[self.mode] + list(sorted(wanted_words))
        if self.BALANCE_DATA:
            self.data_index['training'] = duplicate_data(self.data_index['training'])

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': labels[SILENCE_INDEX],
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index]\
              .extend(unknown_index[set_index][:unknown_size])

        # Custom testing set
        if self.test_dir:
            testing_set = []
            search_path = os.path.join(self.test_dir, '*', '*.wav')
            for wav_path in gfile.glob(search_path):
                _, word = os.path.split(os.path.dirname(wav_path))
                label = optionals[self.mode][UNKNOWN_WORD_INDEX]
                word = word.lower()
                if word in zindexes:
                    l = ([word] + zelements[word])[self.mode % 6]
                    if l in wanted_words:
                        label = l
                    elif self.mode == 6:
                        continue
                testing_set.append({'label': label, 'file': wav_path})
            if testing_set:
                self.data_index['testing'] = testing_set
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = labels
        self.word_to_index = {l: i for i, l in enumerate(labels)}

    def prepare_background_data(self):
        """Searches a folder for background noise audio, and loads it into memory.

        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.

        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.

        Returns:
          List of raw PCM-encoded audio samples of background noise.

        Raises:
          Exception: If files aren't found in the folder.
        """
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data
        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = tf.io.read_file(wav_filename_placeholder)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                       '*.wav')
            for wav_path in gfile.glob(search_path):
                wav_data = sess.run(
                    wav_decoder,
                    feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in '
                                + search_path)

    def prepare_processing_graph(self, model_settings):
        """Builds a TensorFlow graph to apply the input distortions.

        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.

        Args:
          model_settings: Information about the current model being trained.
        """
        g = tf.Graph()
        with g.as_default():
            desired_samples = model_settings['desired_samples']
            self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
            wav_loader = tf.io.read_file(self.wav_filename_placeholder_)
            wav_decoder = tf.audio.decode_wav(
                wav_loader, desired_channels=1, desired_samples=desired_samples)
            # Allow the audio sample's volume to be adjusted.
            self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
            scaled_foreground = tf.multiply(wav_decoder.audio,
                                            self.foreground_volume_placeholder_)
            # Shift the sample's start position, and pad any gaps with zeros.
            self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
            self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
            padded_foreground = tf.pad(
                scaled_foreground,
                self.time_shift_padding_placeholder_,
                mode='CONSTANT')
            sliced_foreground = tf.slice(padded_foreground,
                                         self.time_shift_offset_placeholder_,
                                         [desired_samples, -1])
            self.foreground_volume_placeholder2_ = tf.placeholder(tf.float32, [])
            sliced_foreground = self.foreground_volume_placeholder_ * sliced_foreground
            # Mix in background noise.
            self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                               [desired_samples, 1])
            self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
            background_mul = tf.multiply(self.background_data_placeholder_,
                                         self.background_volume_placeholder_)
            background_add = tf.add(background_mul, sliced_foreground)
            background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
            # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
            import my_signal
            feature_length = model_settings['feature_length']
            self.mfcc_ = my_signal.mfcc(background_clamp,
                                        model_settings['sample_rate'],
                                        frame_length=model_settings['window_size_samples'],
                                        frame_step=model_settings['window_stride_samples'],
                                        num_mfcc=feature_length)
            self.g = g

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_data(self, how_many=100, offset=0, background_frequency=0.0,
                 background_volume_range=0.0, time_shift=0, mode='training', sess=None):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
            how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
            offset: Where to start when fetching deterministically.
            background_frequency: How many clips will have background noise, 0.0 to
            1.0.
            background_volume_range: How loud the background noise will be.
            time_shift: How much to randomly shift the clips by in time.
            mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
            sess: TensorFlow session that was active when processor was created.

        Returns:
            List of sample data for the transformed samples, and list of labels in
            one-hot form.
        """
        close_sess = False
        if not sess:
            sess = tf.Session(graph=self.g, config=self.config)
            close_sess = True

        model_settings = self.model_settings

        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = np.zeros((sample_count, model_settings['fingerprint_size']))
        labels = np.zeros((sample_count, model_settings['label_count']))
        sim_vecs = []
        desired_samples = model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            if pick_deterministically:
                foreground_vol = 1.0
            else:
                foreground_vol = np.random.uniform(0.6, 1.0)
            sample = candidates[sample_index]
            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                # pad / remain head, cut tail
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                # pad / remain tail, cut head
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
                self.time_shift_padding_placeholder_: time_shift_padding,
                self.time_shift_offset_placeholder_: time_shift_offset,
                self.foreground_volume_placeholder2_: foreground_vol,
            }
            # Choose a section of background noise to mix in.
            if use_background:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.random.randint(
                    0, len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(
                    background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples,
                                                                  1])
                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume
            # If we want silence, mute out the main sample but leave the background.
            if sample['label'] == self.words_list[SILENCE_INDEX]:
                input_dict[self.foreground_volume_placeholder_] = 0
            else:
                input_dict[self.foreground_volume_placeholder_] = 1
            # Run the graph to produce the output audio.
            data[i - offset, :] = sess.run(self.mfcc_,
                                           feed_dict=input_dict).flatten()
            label_index = self.word_to_index[sample['label']]
            labels[i - offset, label_index] = 1
            sim_vec = labels[i - offset].tolist()
            if self.sim_mode:
                sim_vec = sim_tables[idx_mode[self.mode]][sample['label']]
                sim_vec = list(map(lambda v: 1.0 if v >= 0.75 else 0.0, sim_vec))
            sim_vecs.append(sim_vec)
        if close_sess:
            sess.close()
        return data, labels, labels

    def get_unprocessed_data(self, how_many, offset, mode, sess=None):
        """Retrieve sample data for the given partition, with no transformations.

        Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the samples, and list of labels in one-hot form.
        """
        results = self.get_data(how_many, offset,
                                0.0, 0.0, 0, mode, sess)
        return results
