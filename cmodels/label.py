import os
from cmodels.idata import rev_zindexes, make_input, optionals

from cmodels import tf

_to_print_results = False if 'to_print_results' in os.environ \
    and os.environ['to_print_results'] == '0' else True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


tf.logging.set_verbosity(tf.logging.ERROR)


class CModel:
    model = None

    def __init__(self, pb_path):

        def _graph_def_to_graph(graph_def):
            tf.reset_default_graph()
            tf.import_graph_def(graph_def, name='')
            return tf.get_default_graph()

        def load_graph_def(pb_path):
            with tf.gfile.FastGFile(pb_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            return graph_def

        if not tf.gfile.Exists(pb_path):
            raise FileNotFoundError('Graph file does not exist %s' % pb_path)
        else:
            fraction = 0.007
            config = None
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                # Assume that you have 12GB of GPU memory and want to allocate ~24MB:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=fraction)
                config = tf.ConfigProto(gpu_options=gpu_options)
            if not self.model:
                print('Loading create syllable graph...')
                self.model = tf.Session(config=config, graph=_graph_def_to_graph(
                    load_graph_def(pb_path)))
                print('Graph(C) Loaded')
                self.rev_zindexes = dict(rev_zindexes)

    def get_labels(self, input_, input_layer_name, output_layer_name):
        sess = self.model
        if sess:
            vector = make_input(input_)
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            predictions, = sess.run(softmax_tensor, {input_layer_name: [vector]})

            s_opt = len(predictions) - len(self.rev_zindexes)
            if 0 < s_opt < 3:
                self.rev_zindexes = {k + s_opt: v for k, v in self.rev_zindexes.items()}
                for i, v in enumerate(optionals[-1][:s_opt]):
                    self.rev_zindexes[i] = v
            elif 0 < -s_opt < 3:
                s_opt = -s_opt
                self.rev_zindexes = {k - s_opt: v for k, v in self.rev_zindexes.items()
                                     if k >= s_opt}
            elif s_opt != 0:
                raise ValueError("output not fit rev zindex")

            # Sort to show labels in order of confidence
            top_k = predictions.argsort()[:][::-1]

            words = []
            words_score = []

            c = 0
            for node_id in top_k:
                human_string = self.rev_zindexes[node_id]
                score = predictions[node_id]
                if c < 9 and _to_print_results:
                    print(human_string, ':', score)
                    c += 1
                words.append(human_string)
                words_score.append(score)

            return words, words_score
        else:
            return

    def is_loaded(self):
        return self.model is not None

    def close(self):
        if self.is_loaded():
            self.model.close()
        self.model = None
