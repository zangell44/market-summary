"""
TODO: Docstring
"""

import json
import os
import numpy as np
import re
import sys
import tensorflow as tf

import model, sample, encoder, webscraping

import datetime
from dateutil import parser
from pandas_datareader import data as pdr
import pymysql


# database connection params
HOST = os.environ.get("HOST")
PORT = int(os.environ.get("PORT"))
DBNAME = os.environ.get("DBNAME")
USER = os.environ.get("USER")
PASSWORD = os.environ.get("PASSWORD")


### Web Scraping ###


def get_stock_data(date, prev):
    """
    Gets information about daily stock market activity and returns a descriptive string

    Parameters
    ----------
    date : str
        Date formatted as YYYY-MM-DD

    Returns
    -------
    summary : str
    """

    # everything below is hardcoded to one timestep

    # get data from date and prior day
    sp500 = pdr.StooqDailyReader("SPY",
                                 start=prev,
                                 end=date).read()

    # get key metrics
    sp500_return = (sp500.iloc[1].Close -
                    sp500.iloc[0].Close) / sp500.iloc[0].Close

    sp500_high = sp500.iloc[1].High * 10
    sp500_low = sp500.iloc[1].Low * 10

    if abs(sp500_return - 0.0) < 0.3:
        return 'The S&P500 Index traded up as high as %.2f, and as low as ' \
               '%.2f, but closed flat, returning %.2f%% on the day.' % (sp500_high, sp500_low, sp500_return)
    elif sp500_return > 0.0:
        return 'The S&P500 Index traded up today, returning %.2f%%. The index traded between' \
               '%.2f and %.2f.' % (sp500_return, sp500_low, sp500_high)
    else:
        return 'The S&P500 Index traded down today, losing %.2f%%. The index traded between' \
               '%.2f and %.2f.' % (sp500_return, sp500_low, sp500_high)


def get_bond_data(date, prev):
    """
    Gets information about daily bond market activity and returns a descriptive string

    Parameters
    ----------
    date : str
        Date formatted as YYYY-MM-DD

    Returns
    -------
    summary : str
    """

    # get bond yields for past two days
    bond_yields = pdr.FredReader('DGS10',
                                 start=prev,
                                 end=date).read()

    curr, prev = bond_yields.iloc[0][0], bond_yields.iloc[1][0]

    if curr > prev:
        return 'US 10 Year Treasury yields rose to %.2f%% from %.2f%% today.' % (curr, prev)

    return 'US 10 Year Treasury yields fell to %.2f%% from %.2f%% today.' % (curr, prev)


def get_commodity_data(date, prev):
    """
    Gets information about daily commodity market activity and returns a descriptive string

    Parameters
    ----------
    date : str
        Date formatted as YYYY-MM-DD

    Returns
    -------
    summary : str
    """
    gold = pdr.FredReader('GOLDAMGBD228NLBM',
                          start=prev,
                          end=date).read()
    g_curr, g_prev = gold.iloc[0][0], gold.iloc[1][0]

    if g_curr > g_prev:
        return 'Gold rose to %.2f from %.2f.' % (g_curr, g_prev)

    return 'Gold fell to %.2f from %.2f.' % (g_curr, g_prev)


def get_daily_activity(date):
    """
    Gets information about daily market activity and returns a descriptive string

    Parameters
    ----------
    date : str
        Date formatted as YYYY-MM-DD

    Returns
    -------
    summary : str
        A string describing the day's activity.
    """
    # check for Monday - if so we return difference from Friday to Monday
    if parser.parse(date).weekday() == 0:
        delta = 3
    else:
        delta = 1

    # get previous trading day
    prev = (parser.parse(date) - datetime.timedelta(delta)).strftime('%Y-%m-%d')

    lead_in = 'Market action in the United States was driven primarily by'

    return ' '.join([get_stock_data(date, prev),
                     get_bond_data(date, prev),
                     get_commodity_data(date, prev),
                     lead_in])


### GPT-2 Model ###

def interact_model(
    raw_text='This is a test',
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=400,
    temperature=1.05,
    top_k=40,
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

                # filter out bad samples
                regex = re.compile('.+?(?=\<\|endoftext\|\>)')
                match = regex.findall(raw_text + text)
                if match:
                    # check if match length is greater than 1000 chars
                    if len(match[0]) > 1000:
                        # append to list
                        samples.append(raw_text + match[0])
                else:
                    # append to output list
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
    prompt = get_daily_activity(date)
    # return a list of possible summaries
    return interact_model(raw_text=prompt, nsamples=int(nsamples))


### MAIN ###

if __name__ == '__main__':
    # generate market summaries
    summaries = generate_samples(date=sys.argv[1], nsamples=sys.argv[2])
    # print (summaries)

    # initialize sql connection
    conn = pymysql.connect(HOST, user=USER, port=PORT,
                           passwd=PASSWORD, db=DBNAME)
    curs = conn.cursor()

    # remove data for given date
    

    # populate data
    insert_query = "INSERT INTO Summary (Date, Commentary) VALUES (%s, %s)"
    # provide a list of tuples to insert
    values = [(sys.argv[1], summary_text) for summary_text in summaries]
    curs.executemany(insert_query, values)

    # close cursor and commit
    conn.commit()
    curs.close()
    conn.close()
    print ('Summaries for date %s inserted succesfully!' % (sys.argv[1]))
