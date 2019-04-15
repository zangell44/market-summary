"""
TODO: Docstring
"""

import datetime
from dateutil import parser
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf


yf.pdr_override()


# TODO
# 1. Make adjustments for trading days (e.g. compare Monday data to Friday close
# 2. Create main function to run scripts


def get_stock_data(date):
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
    yf.pdr_override()

    # everything below is hardcoded to one timestep

    # get data from date and prior day
    sp500 = pdr.get_data_yahoo("^GSPC",
                               start=(parser.parse(date) - datetime.timedelta(1)).strftime('%Y-%m-%d'),
                               end=date)

    # get key metrics
    sp500_return = (sp500.iloc[1].Close -
                    sp500.iloc[0].Close) / sp500.iloc[0].Close

    sp500_high = sp500.iloc[1].High
    sp500_low = sp500.iloc[1].Low

    if abs(sp500_return - 0.0) < 0.3:
        return 'The S&P500 Index traded up as high as %.2f, and as low as ' \
               '%.2f, but closed flat, returning %.2f%% on the day.' % (sp500_high, sp500_low, sp500_return)
    elif sp500_return > 0.0:
        return 'The S&P500 Index traded up today, returning %.2f%%. The index traded between' \
               '%.2f and %.2f.' % (sp500_return, sp500_low, sp500_high)
    else:
        return 'The S&P500 Index traded down today, losing %.2f%%. The index traded between' \
               '%.2f and %.2f.' % (sp500_return, sp500_low, sp500_high)


def get_bond_data(date):
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
                                 start=(parser.parse(date) - datetime.timedelta(1)).strftime('%Y-%m-%d'),
                                 end=date).read()

    curr, prev = bond_yields.iloc[0][0], bond_yields.iloc[1][0]

    if curr > prev:
        return 'US 10 Year Treasury yields rose to %.2f%% from %.2f%% today.' % (curr, prev)

    return 'US 10 Year Treasury yields fell to %.2f%% from %.2f%% today.' % (curr, prev)


def get_commodity_data(date):
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
                          start=(parser.parse(date) - datetime.timedelta(1)).strftime('%Y-%m-%d'),
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
    return ' '.join([get_stock_data(date),
                     get_bond_data(date),
                     get_commodity_data(date)])