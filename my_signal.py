import tensorflow as _tf

_major, _minor, *_ = [int(v) for v in _tf.__version__.split('.')[:2]]
feature_length = 40

def spectrogram(signal, frame_length=512, frame_step=128, fft_length=None,
                window_fn=_tf.signal.hann_window, power=2.0, **kwargs):

    signal = _tf.transpose(signal)
    stfts = _tf.signal.stft(signal,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           fft_length=fft_length,
                           window_fn=window_fn,
                           **kwargs)
    magnitude = _tf.abs(stfts)

    return _tf.pow(magnitude, _tf.constant(power, dtype=_tf.float32))


def rosa_mel(sr, num_spectrogram_bins, n_mels=128, fmin=20.0, fmax=4000.0, **kwargs):

    from librosa.filters import mel
    n_fft = int(num_spectrogram_bins - 1) * 2
    mel_basis = mel(sr, n_fft, n_mels,fmin, fmax, **kwargs)
    return _tf.constant(mel_basis.transpose())


def tf_mel(sr, num_spectrogram_bins, n_mels=80, fmin=20.0, fmax=None, **kwargs):
    if fmax is None:
        fmax = float(sr) / 2
    return _tf.signal.linear_to_mel_weight_matrix(n_mels, num_spectrogram_bins,
                                                 sr, fmin, fmax, **kwargs)


def power_to_db(tensor, amin=1e-20, ref_value=1.0, top_db=80.0):

    from math import log

    base = 10.0
    scalar = _tf.constant(base / log(base), _tf.float32)
    ref_value = _tf.abs(ref_value)
    magnitude = _tf.abs(tensor)
    log_spec = scalar * _tf.math.log(magnitude + amin)
    log_spec -= scalar * _tf.math.log(ref_value)

    if top_db is not None:
        if top_db < 0:
            raise ValueError('top_db must be non-negative')
        log_spec = _tf.maximum(log_spec, _tf.reduce_max(log_spec) - top_db)
    return log_spec


def mel_spectrogram_l(signal, sr, S=None, frame_length=512, frame_step=128,
         fft_length=None, window_fn=_tf.signal.hamming_window,
         power=2.0, types='rosa', name='mel_spectrogram', **kwargs):

    if S is None:
        S = spectrogram(signal, frame_length, frame_step, fft_length,
                        window_fn, power)

    num_spectrogram_bins = S.shape[-1]
    if types == 'rosa':
        mel_basis = rosa_mel(sr, num_spectrogram_bins, **kwargs)
        mel_spectrogram = _tf.tensordot(S, mel_basis, 1)

        mel_spectrogram.set_shape(S.shape[:-1].concatenate(mel_basis.shape[-1:]))

        log_mel_spectrogram = power_to_db(mel_spectrogram)

    else:
        mel_basis = tf_mel(sr, num_spectrogram_bins, **kwargs)
        mel_spectrogram = _tf.tensordot(S, mel_basis, 1)

        mel_spectrogram.set_shape(S.shape[:-1].concatenate(mel_basis.shape[-1:]))

        log_mel_spectrogram = _tf.math.log(mel_spectrogram)

    return log_mel_spectrogram


def mfcc(signal, sr, S=None, frame_length=512, frame_step=128,
         fft_length=None, window_fn=_tf.signal.hamming_window,
         power=2.0, num_mfcc=13, types='rosa', name='mfcc', **kwargs):

    if S is None:
        S = spectrogram(signal, frame_length, frame_step, fft_length,
                        window_fn, power)

    num_spectrogram_bins = S.shape[-1]
    if types == 'rosa':
        mel_basis = rosa_mel(sr, num_spectrogram_bins, **kwargs)
        mel_spectrogram = _tf.tensordot(S, mel_basis, 1)

        mel_spectrogram.set_shape(S.shape[:-1].concatenate(mel_basis.shape[-1:]))

        log_mel_spectrogram = power_to_db(mel_spectrogram)
        mfccs = _tf.signal.dct(
            log_mel_spectrogram, norm='ortho', name=name)[..., :num_mfcc]

        return mfccs
    else:
        mel_basis = tf_mel(sr, num_spectrogram_bins, **kwargs)
        mel_spectrogram = _tf.tensordot(S, mel_basis, 1)

        mel_spectrogram.set_shape(S.shape[:-1].concatenate(mel_basis.shape[-1:]))

        log_mel_spectrogram = _tf.math.log(mel_spectrogram)
        scalars = [pow(2., 0.5)] + [1.0] * (num_mfcc - 1)
        s = _tf.constant(scalars, _tf.float32)
        mfccs = s * _tf.signal.dct(
            log_mel_spectrogram, norm='ortho', name=name)[..., :num_mfcc]

        return mfccs


def decode_wav(f):
    wav_loader = _tf.io.read_file(f)
    wav_decoder = _tf.audio.decode_wav(
        wav_loader, 1)
    return wav_decoder
