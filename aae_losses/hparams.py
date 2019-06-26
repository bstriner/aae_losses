import json
import os

import six
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.contrib.training import HParams

HPARAMS_FILE = 'configuration.json'


def load_hparams(model_dir, default_params):
    hparams_path = os.path.join(model_dir, HPARAMS_FILE)
    hparams = HParams(default_params)
    assert os.path.exists(hparams_path)
    with open(hparams_path) as f:
        return hparams.parse_json(f.read())


def write_hparams_events(model_dir, hparams):
    # Write HParams events
    hparams_dict = hparams.values()
    hparams_dict_train = hparams_dict.copy()
    hparams_dict_eval = hparams_dict.copy()
    hparams_dict_train['mode'] = 'train'
    hparams_dict_eval['mode'] = 'eval'
    hparams_pb = hp.hparams_pb(hparams_dict_train).SerializeToString()
    hparams_pb_eval = hp.hparams_pb(hparams_dict_eval).SerializeToString()
    with tf.summary.FileWriter(model_dir) as w:
        w.add_summary(hparams_pb)
    with tf.summary.FileWriter(os.path.join(model_dir, 'eval')) as w:
        w.add_summary(hparams_pb_eval)


def write_hparams_json(hparams_path, hparams):
    with open(hparams_path, 'w') as f:
        json.dump(obj=hparams.values(), fp=f, sort_keys=True, indent=4)


def validate_hparams(hparams_path, hparams):
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparam_dict = json.load(f)
        for k, v in six.iteritems(hparam_dict):
            oldval = getattr(hparams, k)
            assert oldval == v, "Incompatible key {}: save {}-> config {}".format(k, oldval, v)


def create_hparams(default_params, hparams_file=None, hparams_str=None):
    hparams = HParams(**default_params)
    if hparams_file is not None:
        with open(hparams_file) as f:
            hparams.parse_json(f.read())
    if hparams_str is not None:
        hparams.parse(hparams_str)
    return hparams


def get_hparams(model_dir, default_params, hparams_file=None, hparams_str=None, validate=True):
    hparams_path = os.path.join(model_dir, HPARAMS_FILE)

    # Load HParams
    hparams = create_hparams(
        default_params=default_params,
        hparams_file=hparams_file,
        hparams_str=hparams_str)

    # Validate HParams
    if validate:
        validate_hparams(hparams_path, hparams)

    # Write HParams file
    write_hparams_json(
        hparams_path=hparams_path,
        hparams=hparams)

    # Write HParams events
    write_hparams_events(
        model_dir=model_dir,
        hparams=hparams)

    return hparams
