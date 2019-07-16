import numpy as np
import tensorflow as tf
import logging
import functools
import os
import sys
import json
from src import model, sample, encoder

class gpt2_server_sessions:

    def __init__(self,server_id):
        self.server_id = server_id
        self.conf_path = os.path.join('config', 'servers')
        self.load_json(server_id)
        json_conf = self.server_configs
        self.init_state(json_conf['nsamples'],json_conf['length'],json_conf['temperature'],json_conf['top_k'],json_conf['model_name'])
        self.reset_model()

    def init_state(self, nsamples=1, length=200, temperature=1, top_k=0, model_name='117M'):
        self.model_name = model_name
        self.batch_size = 1
        self.seed = 42069
        self.nsamples = nsamples
        self.length = length
        self.temperature = temperature
        self.top_k = top_k

    def set_state(self, nsamples, length, temperature, top_k, model_name='117M'):
        self.nsamples = nsamples
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        self.model_name = model_name
        self.server_configs['model_name'] = model_name
        self.server_configs['nsamples'] = nsamples
        self.server_configs['length'] = length
        self.server_configs['top_k'] = top_k
        self.server_configs['temperature'] = temperature
        self.writeConfig(self.server_id)

    def preinit_model(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.enc = encoder.get_encoder(self.model_name)
        self.hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
        elif self.length > self.hparams.n_ctx:
            logging.error("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

    def init_model(self):
        self.context = tf.placeholder(tf.int32, [1, None])
        self.output = sample.sample_sequence(
            hparams=self.hparams, length=self.length,
            context=self.context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k
        )

        self.saver = tf.train.Saver()
        self.ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
        self.saver.restore(self.session, self.ckpt)
        self.is_inferencing = False

    def reset_model(self):
        self.init_state(self.server_configs['nsamples'],self.server_configs['length'],self.server_configs['temperature'],self.server_configs['top_k'],self.server_configs['model_name'])
        self.preinit_model()
        self.session = tf.InteractiveSession(graph=tf.Graph())
        self.init_model()

    def shutdown(self):
        logging.info('Shutting down GPT.')
        self.session.close()

    def writeConfig(self,server_id):
        with open(os.path.join(self.conf_path, str(server_id) + ".json"), "w", encoding='utf-8') as f:
            f.write(json.dumps(self.server_configs, indent=3, ensure_ascii=False))

    def load_json(self, server_id):
        filename = os.path.join(self.conf_path,str(server_id)+".json")
        if os.path.isfile(filename):
            self.server_configs = json.load(filename)
        else:
            self.server_configs = self.default_config()

    def default_config(self):
        return {
        'model_name':'117M',
        'nsamples':1,
        'length':200,
        'temperature':1,
        'top_k':40
        }
    def generate_text(self, context_tokens):
        return self.session.run(self.output, feed_dict={
                    self.context: [context_tokens for _ in range(1)]
                })[:, len(context_tokens):]