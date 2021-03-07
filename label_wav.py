import os

import tensorflow as tf
tf = tf.compat.v1

_to_print_results = False if 'to_print_results' in os.environ \
    and os.environ['to_print_results'] == '0' else True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.logging.set_verbosity(tf.logging.ERROR)


def _graph_def_to_graph(graph_def):
    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')
    return tf.get_default_graph()


def load_graph_def(pb_path):
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graphs_def(all_path):
    all_graph_def = [load_graph_def(p) for p in all_path]
    return all_graph_def


def load_models(all_path):
    graph_defs = load_graphs_def(all_path)
    all_graph = []
    config = None
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        fraction = 0.04 * len(graph_defs)
        # Assume that you have 12GB of GPU memory and want to allocate ~24MB:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True

    if not all_graph:
        all_graph = [_graph_def_to_graph(a) for a in graph_defs]
    try:
        models = [tf.Session(graph=g, config=config) for g in all_graph]
        return models
    except Exception:
        print('model loading unsuccessfully')
        return None


def run_graph(wav_data, labels, input_layer_name, output_layer_name, sess, suffix='',
              softmax_tensor=None):
    """Runs the audio data through the graph and prints predictions."""
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    if softmax_tensor is None:
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[:][::-1]

    words = []
    words_score = []

    for i, node_id in enumerate(top_k):
        human_string = labels[node_id]
        if '_' in human_string:
            human_string += suffix  # no_label[1-3]
        score = predictions[node_id]

        if i < 9 and _to_print_results:
            print(human_string, ':', score)

        words.append(human_string)
        words_score.append(score)

    return words, words_score, softmax_tensor


class AcousticModels:
    loaded = False

    def __init__(self, model_paths, label_paths):

        self.labels = [load_labels(p) for p in label_paths]
        self.models = load_models(model_paths)
        self.tensors = [None for _ in model_paths]
        self.loaded = True

    def is_loaded(self):
        return self.loaded

    def close(self):
        for sess in self.models:
            sess.close()
        self.models = []
        self.labels = []
        self.loaded = False

    def label_wav(self,
                  wav,
                  input_name='file',
                  output_name='labels_softmax:0',
                  mode=0,
                  suffix=''):
        if input_name == 'file' and (not wav or not tf.gfile.Exists(wav)):
            tf.logging.fatal('Audio file does not exist %s', wav)

        if -1 < mode < len(self.labels):
            labels_list = self.labels[mode]
            sess = self.models[mode]
            tensor = self.tensors[mode]
            wav_data = wav
            if input_name == 'file':
                with open(wav, 'rb') as wav_file:
                    wav_data = wav_file.read()
                input_name = 'wav_data:0'
            *res, t = run_graph(wav_data, labels_list, input_name, output_name, sess, suffix, tensor)
            if tensor is None:
                self.tensors[mode] = t
            return res
        else:
            raise IndexError('Out of Range')
