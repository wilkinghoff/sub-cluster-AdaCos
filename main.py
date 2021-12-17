import pandas as pd
import numpy as np
import keras
import os
import soundfile as sf
import tensorflow as tf
import librosa
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import class_weight
import common as com
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
from subcluster_adacos import SCAdaCos
from sklearn.mixture import GaussianMixture


def mixupLoss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true=y_pred[:, :, 1], y_pred=y_pred[:, :, 0])


def make_mean(mat, label):
    label, index = np.unique(label, return_inverse=True)
    mean = []
    mat = np.array(mat)
    for i, spk in enumerate(label):
        mean.append(np.mean(mat[np.nonzero(index == i)], axis=0))
    mean = length_norm(mean)
    return mean, label


def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


class LogMelExtractor(object):
    """
    Original source code (before changes): https://github.com/qiuqiangkong/dcase2019_task1
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        """Log mel feature extractor.

        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        """

        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)

        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax).T

    def transform(self, audio):
        """Extract feature of a singlechannel audio file.

        Args:
          audio: (samples,)

        Returns:
          feature: (frames_num, freq_bins)
        """

        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func

        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio,
            n_fft=window_size,
            hop_length=hop_size,
            window=window_func,
            center=True,
            dtype=np.complex64,
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''

        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)

        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10,
            top_db=None)

        logmel_spectrogram = logmel_spectrogram.astype(np.float32).transpose()
        return logmel_spectrogram


def model_cnn(emb_size, num_classes, time_dim, min_val, n_subclusters):
    data_input = tf.keras.layers.Input(shape=(time_dim, emb_size, 1), dtype='float32')
    label_input = tf.keras.layers.Input(shape=(num_classes), dtype='float32')
    y = label_input
    x = data_input
    l2_weight_decay = tf.keras.regularizers.l2(1e-5)
    x, y = MixupLayer(prob=1)([x, y])

    # first block
    x = tf.keras.layers.Conv2D(16, 7, strides=2, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    # second block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    # third block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=32, strides=1, padding="same", kernel_regularizer=l2_weight_decay)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    # fourth block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=64, strides=1, padding="same", kernel_regularizer=l2_weight_decay)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    # fifth block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding="same", kernel_regularizer=l2_weight_decay)(x)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)

    # get embeddings and classify
    x = tf.keras.layers.MaxPooling2D((10, 1), padding='same')(x)
    x = tf.keras.layers.Flatten(name='flat')(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, name='emb')(x)
    output = SCAdaCos(n_classes=num_classes, n_subclusters=n_subclusters)([x, y, label_input])
    loss_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([output, y])

    return data_input, label_input, loss_output


########################################################################################################################
# Load data and compute embeddings
########################################################################################################################
# emb_size = 6144
n_log_mel = 128
target_sr = 16000
param = com.yaml_load()
extractor = LogMelExtractor(target_sr, 1024, 512, mel_bins=n_log_mel, fmin=0, fmax=target_sr/2)

# load train data
print('Loading train data')
categories = os.listdir("./dev_data")

if os.path.isfile(str(n_log_mel) + '_train_log_mel.npy'):
    train_log_mel = np.load(str(n_log_mel) + '_train_log_mel.npy')
    train_ids = np.load('train_ids.npy')
    train_files = np.load('train_files.npy')
else:
    train_log_mel = []
    train_ids = []
    train_files = []
    dicts = ['./dev_data/', './eval_data/']
    #dicts = ['./eval_data/']
    #dicts = ['./dev_data/']
    eps=1e-12
    for label, category in enumerate(categories):
        print(category)
        for dict in dicts:
            for count, file in tqdm(enumerate(os.listdir(dict + category + "/train")), total=len(os.listdir(dict + category + "/train"))):
                file_path = dict + category + "/train/" + file
                wav, fs = sf.read(file_path)
                wav = librosa.core.to_mono(wav.transpose()).transpose()
                # extract log_mels
                log_mel = extractor.transform(wav).transpose()
                if log_mel.shape[0] > 313:
                    log_mel = log_mel[log_mel.shape[0]-313:, :]
                train_log_mel.append(log_mel)
                train_ids.append(category + '_' + file.split('_')[-2])
                train_files.append(file_path)
    # reshape arrays and store
    train_ids = np.array(train_ids)
    train_files = np.array(train_files)
    train_log_mel = np.expand_dims(np.array(train_log_mel, dtype=np.float32), axis=-1)
    np.save('train_ids.npy', train_ids)
    np.save('train_files.npy', train_files)
    np.save(str(n_log_mel) + '_train_log_mel.npy', train_log_mel)

# load evaluation data
print('Loading evaluation data')
if os.path.isfile(str(n_log_mel) + '_eval_log_mel.npy'):
    eval_log_mel = np.load(str(n_log_mel) + '_eval_log_mel.npy')
    eval_ids = np.load('eval_ids.npy')
    eval_normal = np.load('eval_normal.npy')
    eval_files = np.load('eval_files.npy')
else:
    eval_log_mel = []
    eval_ids = []
    eval_normal = []
    eval_files = []
    eps=1e-12
    for label, category in enumerate(categories):
        print(category)
        for count, file in tqdm(enumerate(os.listdir("./dev_data/" + category + "/test")), total=len(os.listdir("./dev_data/" + category + "/test"))):
            file_path = "./dev_data/" + category + "/test/" + file
            wav, fs = sf.read(file_path)
            wav = librosa.core.to_mono(wav.transpose()).transpose()
            # extract log_mels
            log_mel = extractor.transform(wav).transpose()
            if log_mel.shape[0] > 313:
                log_mel = log_mel[log_mel.shape[0]-313:, :]
            eval_log_mel.append(log_mel)
            eval_ids.append(category + '_' + file.split('_')[-2])
            eval_normal.append(file.split('_')[0] == 'normal')
            eval_files.append(file_path)
    # reshape arrays and store
    eval_ids = np.array(eval_ids)
    eval_normal = np.array(eval_normal)
    eval_files = np.array(eval_files)
    eval_log_mel = np.expand_dims(np.array(eval_log_mel, dtype=np.float32), axis=-1)
    np.save('eval_ids.npy', eval_ids)
    np.save('eval_normal.npy', eval_normal)
    np.save('eval_files.npy', eval_files)
    np.save(str(n_log_mel) + '_eval_log_mel.npy', eval_log_mel)

# load test data
print('Loading test data')
if os.path.isfile(str(n_log_mel) + '_test_log_mel.npy'):
    test_log_mel = np.load(str(n_log_mel) + '_test_log_mel.npy')
    test_ids = np.load('test_ids.npy')
    test_files = np.load('test_files.npy')
else:
    test_log_mel = []
    test_ids = []
    test_files = []
    eps = 1e-12
    for label, category in enumerate(categories):
        print(category)
        for count, file in tqdm(enumerate(os.listdir("./eval_data/" + category + "/test")), total=len(os.listdir("./eval_data/" + category + "/test"))):
            file_path = "./eval_data/" + category + "/test/" + file
            wav, fs = sf.read(file_path)
            wav = librosa.core.to_mono(wav.transpose()).transpose()
            # extract log_mels
            log_mel = extractor.transform(wav).transpose()
            if log_mel.shape[0] > 313:
                log_mel = log_mel[log_mel.shape[0]-313:, :]
            test_log_mel.append(log_mel)
            test_ids.append(category + '_' + file.split('_')[-2])
            test_files.append(file_path)
    # reshape arrays and store
    test_ids = np.array(test_ids)
    test_files = np.array(test_files)
    test_log_mel = np.expand_dims(np.array(test_log_mel, dtype=np.float32), axis=-1)
    np.save('test_ids.npy', test_ids)
    np.save('test_files.npy', test_files)
    np.save(str(n_log_mel) + '_test_log_mel.npy', test_log_mel)

# encode ids as labels
le = LabelEncoder()
train_labels = le.fit_transform(train_ids)
eval_labels = le.transform(eval_ids)
test_labels = le.transform(test_ids)

# distinguish between normal and anomalous samples
unknown_log_mel = eval_log_mel[~eval_normal]
unknown_labels = eval_labels[~eval_normal]
unknown_files = eval_files[~eval_normal]
unknown_ids = eval_ids[~eval_normal]
eval_log_mel = eval_log_mel[eval_normal]
eval_labels = eval_labels[eval_normal]
eval_files = eval_files[eval_normal]
eval_ids = eval_ids[eval_normal]

# set up dict to convert machine type into a vector indicating all ids that belong to that type
type_dict = {'ToyCar': np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
             'ToyConveyor': np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
             'fan': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
             'pump': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
             'slider': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0]),
             'valve': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1])}
type_dict_s = {'ToyCar': np.array([1,0,0,0,0,0]),
             'ToyConveyor': np.array([0,1,0,0,0,0]),
             'fan': np.array([0,0,1,0,0,0]),
             'pump': np.array([0,0,0,1,0,0]),
             'slider': np.array([0,0,0,0,1,0]),
             'valve': np.array([0,0,0,0,0,1])}

type_labels_train = np.array([type_dict[train_id.split('_')[0]] for train_id in train_ids])
type_labels_eval = np.array([type_dict[eval_id.split('_')[0]] for eval_id in eval_ids])
type_labels_unknown = np.array([type_dict[unknown_id.split('_')[0]] for unknown_id in unknown_ids])
type_labels_train_s = np.array([type_dict_s[train_id.split('_')[0]] for train_id in train_ids])
type_labels_eval_s = np.array([type_dict_s[eval_id.split('_')[0]] for eval_id in eval_ids])
type_labels_unknown_s = np.array([type_dict_s[unknown_id.split('_')[0]] for unknown_id in unknown_ids])

########################################################################################################################
# Preprocessing
########################################################################################################################

# feature normalization
print('Normalizing data')
eps = 1e-12
mean_log_mel = np.expand_dims(np.repeat(np.expand_dims(np.mean(train_log_mel.reshape(train_log_mel.shape[0]*train_log_mel.shape[1], train_log_mel.shape[2], 1), axis=0), axis=0),
                     repeats=train_log_mel.shape[1], axis=0), axis=0)
std_log_mel = np.expand_dims(np.repeat(np.expand_dims(np.std(train_log_mel.reshape(train_log_mel.shape[0]*train_log_mel.shape[1], train_log_mel.shape[2], 1), axis=0), axis=0),
                    repeats=train_log_mel.shape[1], axis=0), axis=0)
train_log_mel = (train_log_mel-mean_log_mel)/(std_log_mel+eps)
eval_log_mel = (eval_log_mel-mean_log_mel)/(std_log_mel+eps)
unknown_log_mel = (unknown_log_mel-mean_log_mel)/(std_log_mel+eps)
test_log_mel = (test_log_mel-mean_log_mel)/(std_log_mel+eps)

########################################################################################################################
# train x-vector cnn on train partition of development set
########################################################################################################################
batch_size = 64
batch_size_test = 64
epochs = 1
aeons = 1
alpha = 1

# predicting with GMMs
pred_eval = np.zeros((eval_log_mel.shape[0], np.unique(train_labels).shape[0], 3))
pred_unknown = np.zeros((unknown_log_mel.shape[0], np.unique(train_labels).shape[0], 3))
pred_test = np.zeros((test_log_mel.shape[0], np.unique(train_labels).shape[0], 3))
pred_train = np.zeros((train_log_mel.shape[0], np.unique(train_labels).shape[0], 3))
for n_subclusters in 2**np.arange(7):
    y_train_cat = keras.utils.np_utils.to_categorical(train_labels, num_classes=len(np.unique(train_labels)))
    y_eval_cat = keras.utils.np_utils.to_categorical(eval_labels, num_classes=len(np.unique(train_labels)))
    y_unknown_cat = keras.utils.np_utils.to_categorical(unknown_labels, num_classes=len(np.unique(train_labels)))

    # compile model
    data_input, label_input, loss_output = model_cnn(emb_size=train_log_mel.shape[2],
                                                             num_classes=len(np.unique(train_labels)),
                                                             time_dim=train_log_mel.shape[1], min_val=np.min(train_log_mel), n_subclusters=n_subclusters)
    model = tf.keras.Model(inputs=[data_input, label_input], outputs=[loss_output])
    model.compile(loss=[mixupLoss], optimizer=tf.keras.optimizers.Adam())
    print(model.summary())
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs"), histogram_freq=0, write_graph=True,
                                       write_images=False)
    ]

    for k in np.arange(aeons):
        print('subclusters: ' + str(n_subclusters))
        print('aeon: ' + str(k))
        # fit model
        weight_path = './models/wts_log_mel_' + str(k + 1) + 'k_' + str(n_log_mel) + '_' + str(n_subclusters) + '.h5'
        if not os.path.isfile(weight_path):
            class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
            class_weights = {i: class_weights[i] for i in range(class_weights.shape[0])}
            model.fit([train_log_mel, y_train_cat, type_labels_train, type_labels_train_s], y_train_cat, verbose=1,
                                batch_size= batch_size, epochs=epochs, callbacks=callbacks,
                                validation_data=([eval_log_mel, y_eval_cat], y_eval_cat), class_weight=class_weights)
            model.save(weight_path)
        else:
            model = tf.keras.models.load_model(weight_path,
                                               custom_objects={'MixupLayer': MixupLayer, 'mixupLoss': mixupLoss, 'SCAdaCos': SCAdaCos})

    emb_model = tf.keras.Model(model.input, model.get_layer('emb').output)
    eval_embs = emb_model.predict([eval_log_mel, y_eval_cat], batch_size=batch_size)
    train_embs = emb_model.predict([train_log_mel, y_train_cat], batch_size=batch_size)
    unknown_embs = emb_model.predict([unknown_log_mel, np.zeros((unknown_log_mel.shape[0], len(np.unique(train_labels))))], batch_size=batch_size)
    test_embs = emb_model.predict([test_log_mel, np.zeros((test_log_mel.shape[0], len(np.unique(train_labels))))], batch_size=batch_size)

    # length normalization
    print('normalizing lengths')
    x_train_ln = length_norm(train_embs)
    x_eval_ln = length_norm(eval_embs)
    x_test_ln = length_norm(test_embs)
    x_unknown_ln = length_norm(unknown_embs)

    model_means = model.layers[-2].get_weights()[0].transpose()
    model_means_ln = length_norm(model_means)

    x_train_ln = np.concatenate([x_train_ln, np.mean(train_log_mel, axis=1)[:, :, 0], np.max(train_log_mel, axis=1)[:, :, 0]], axis=-1)
    x_eval_ln = np.concatenate([x_eval_ln, np.mean(eval_log_mel, axis=1)[:, :, 0], np.max(eval_log_mel, axis=1)[:, :, 0]], axis=-1)
    x_test_ln = np.concatenate([x_test_ln, np.mean(test_log_mel, axis=1)[:, :, 0], np.max(test_log_mel, axis=1)[:, :, 0]], axis=-1)
    x_unknown_ln = np.concatenate([x_unknown_ln, np.mean(unknown_log_mel, axis=1)[:, :, 0], np.max(unknown_log_mel, axis=1)[:, :, 0]], axis=-1)
    for j, lab in tqdm(enumerate(np.unique(train_labels)), total=len(np.unique(train_labels))):
        clf1 = GaussianMixture(n_components=n_subclusters, covariance_type='full', reg_covar=1e-3, means_init=model_means_ln[j * n_subclusters:(j + 1) * n_subclusters]).fit(
            x_train_ln[train_labels == lab, :train_embs.shape[1]])
        clf2 = GaussianMixture(n_components=1, covariance_type='full', reg_covar=1e-3).fit(
            x_train_ln[train_labels == lab, train_embs.shape[1]:train_embs.shape[1] + train_log_mel.shape[2]])
        clf3 = GaussianMixture(n_components=1, covariance_type='full', reg_covar=1e-3).fit(
            x_train_ln[train_labels == lab, train_embs.shape[1] + train_log_mel.shape[2]:])

        pred_eval[:, j, 0] += -np.max(clf1._estimate_log_prob(x_eval_ln[:, :eval_embs.shape[1]]), axis=-1)
        pred_eval[:, j, 1] += -clf2.score_samples(
            x_eval_ln[:, eval_embs.shape[1]:eval_embs.shape[1] + eval_log_mel.shape[2]])
        pred_eval[:, j, 2] += -clf3.score_samples(x_eval_ln[:, eval_embs.shape[1] + eval_log_mel.shape[2]:])

        pred_unknown[:, j, 0] += -np.max(clf1._estimate_log_prob(x_unknown_ln[:, :unknown_embs.shape[1]]), axis=-1)
        pred_unknown[:, j, 1] += -clf2.score_samples(
            x_unknown_ln[:, unknown_embs.shape[1]:unknown_embs.shape[1] + unknown_log_mel.shape[2]])
        pred_unknown[:, j, 2] += -clf3.score_samples(
            x_unknown_ln[:, unknown_embs.shape[1] + unknown_log_mel.shape[2]:])

        pred_test[:, j, 0] += -np.max(clf1._estimate_log_prob(x_test_ln[:, :test_embs.shape[1]]), axis=-1)
        pred_test[:, j, 1] += -clf2.score_samples(
            x_test_ln[:, test_embs.shape[1]:test_embs.shape[1] + test_log_mel.shape[2]])
        pred_test[:, j, 2] += -clf3.score_samples(x_test_ln[:, test_embs.shape[1] + test_log_mel.shape[2]:])

    # use mean for machine type ToyConveyor
    pred_eval_final = pred_eval[:, :, 0]
    pred_unknown_final = pred_unknown[:, :, 0]
    pred_test_final = pred_test[:, :, 0]
    for lab in np.unique(train_labels):
        if le.inverse_transform([lab])[0] == 'ToyConveyor':
            pred_eval_final[:, lab] = pred_eval[:, :, 1]
            pred_unknown_final[:, lab] = pred_unknown[:, :, 1]
            pred_test_final[:, lab] = pred_test[:, :, 1]

    # output performance
    print('performance on evaluation set')
    y_pred_eval = np.argmin(pred_eval_final, axis=1)
    y_pred_unknown = np.argmin(pred_unknown_final, axis=1)
    print('####################')
    print('closed-set performance by machine id:')
    print('evaluation files: ' + str(np.mean(y_pred_eval == eval_labels)))
    print('unknown files: ' + str(np.mean(y_pred_unknown == unknown_labels)))
    print('all files: ' + str(
        np.mean(np.hstack([y_pred_unknown, y_pred_eval]) == np.hstack([unknown_labels, eval_labels]))))
    print('####################')
    type_labels_eval1 = np.array([eval_id.split('_')[0] for eval_id in eval_ids])
    type_labels_unknown1 = np.array([unknown_id.split('_')[0] for unknown_id in unknown_ids])
    type_pred_eval1 = np.array([pred_id.split('_')[0] for pred_id in le.inverse_transform(y_pred_eval)])
    type_pred_unknown1 = np.array([pred_id.split('_')[0] for pred_id in le.inverse_transform(y_pred_unknown)])
    print('closed-set performance by machine type:')
    print('evaluation files: ' + str(np.mean(type_pred_eval1 == type_labels_eval1)))
    print('unknown files: ' + str(np.mean(type_pred_unknown1 == type_labels_unknown1)))
    print('all files: ' + str(np.mean(
        np.hstack([type_pred_unknown1, type_pred_eval1]) == np.hstack([type_labels_unknown1, type_labels_eval1]))))
    print('####################')
    print('closed-set performance on test data')
    y_pred_test = np.argmin(pred_test_final, axis=1)
    type_labels_test1 = np.array([test_id.split('_')[0] for test_id in test_ids])
    type_pred_test1 = np.array([pred_id.split('_')[0] for pred_id in le.inverse_transform(y_pred_test)])
    print('for machine id: ' + str(np.mean(y_pred_test == test_labels)))
    print('for machine type: ' + str(np.mean(type_pred_test1 == type_labels_test1)))
    print('####################')
    aucs = []
    p_aucs = []
    for j, cat in enumerate(np.unique(eval_ids)):
        y_pred = np.concatenate([pred_eval_final[eval_labels == le.transform([cat]), le.transform([cat])],
                                 pred_unknown_final[unknown_labels == le.transform([cat]), le.transform([cat])]],
                                axis=0)
        y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                                 np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
        p_auc = roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
        p_aucs.append(p_auc)
        print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
        print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))
    print('####################')
    aucs = np.array(aucs)
    p_aucs = np.array(p_aucs)
    for cat in categories:
        mean_auc = np.mean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
        mean_p_auc = np.mean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
    print('####################')
    for cat in categories:
        mean_auc = np.mean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        mean_p_auc = np.mean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
        print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
    print('####################')
    mean_auc = np.mean(aucs)
    print('mean AUC: ' + str(mean_auc * 100))
    mean_p_auc = np.mean(p_aucs)
    print('mean pAUC: ' + str(mean_p_auc * 100))

    # create challenge submission files
    print('creating submission files')
    for j, cat in enumerate(np.unique(test_ids)):
        file_idx = test_labels == le.transform([cat])
        results = pd.DataFrame()
        results['output1'], results['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                  [str(s) for s in pred_test_final[file_idx, le.transform([cat])]]]
        results.to_csv('teams/mfcc_emb/anomaly_score_' + cat.split('_')[0] + '_id_' + cat.split('_')[-1] + '.csv',
                       encoding='utf-8', index=False, header=False)
    print('####################')
    print('>>>> finished! <<<<<')
    print('####################')
