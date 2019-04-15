#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import sys
import tensorflow as tf

import model, sample, encoder, webscraping

def interact_model(
    raw_text='This is a test',
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=300,
    temperature=1,
    top_k=0,
):
    """
    Run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """

    samples = []

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                samples.append(raw_text + text)

    return samples

def generate_samples(
        date,
        nsamples
):
    """
    Generates text samples for a given date

    :param date: str
        Date string of the form YYYY-MM-DD

    :param nsamples: int
        Number of text samples to generate

    :return: list
        List of generated text
    """

    # get seed statement based on market activity
    prompt = webscraping.get_daily_activity(date)
    # return a list of possible summaries
    return interact_model(raw_text=prompt, nsamples=nsamples)


if __name__ == '__main__':
    generate_samples(date=sys.argv[1], nsamples=sys.argv[2])
