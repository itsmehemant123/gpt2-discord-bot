import json
import os
import model, sample, encoder
import numpy as np
import tensorflow as tf

# import gpt.model, gpt.sample, gpt.encoder

model_name='117M'
batch_size = 1
seed = None
nsamples=1
length=10
temperature=1
top_k=0
np.random.seed(seed)
tf.set_random_seed(seed)

enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

with tf.Session(graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [1, None])
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=1,
        temperature=temperature, top_k=top_k
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    saver.restore(sess, ckpt)

    raw_text = "hello sir"
    context_tokens = enc.encode(raw_text)
    generated = 0
    out = sess.run(output, feed_dict={
        context: [context_tokens for _ in range(1)]
    })[:, len(context_tokens):]
    generated += 1
    text = enc.decode(out[0])
    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
    print(text)
    print("=" * 80)
