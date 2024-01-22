import os
import numpy as np
import yfinance as yf
import psycopg2
from psycopg2 import pool
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
db_name = os.getenv('name')
db_user = os.getenv('user')
db_password = os.getenv('pass')
db_host = os.getenv('host', 'localhost')
db_port = os.getenv('port', '5432')
conn_pool = psycopg2.pool.SimpleConnectionPool(1, 10, dbname=db_name, user=db_user, password=db_password, host=db_host,
                                               port=db_port)


def american_option_binomial_tree(S, K, T, r, sigma, is_call=True, N=100):
    """
    Price an option using the Binomial Tree model.

    S: current stock price
    K: Strike price
    T: Time to maturity in years
    r: Risk-free rate
    sigma: Volatility
    is_call: True for call option, False for put option
    N: Number of steps in the binomial tree
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    ST = np.zeros((N + 1, N + 1))
    ST[0, 0] = S
    for i in range(1, N + 1):
        ST[i, 0] = ST[i - 1, 0] * u
        for j in range(1, i + 1):
            ST[i, j] = ST[i - 1, j - 1] * d

    if is_call:
        option = np.maximum(ST - K, 0)
    else:
        option = np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[i, j] = max(option[i + 1, j] * p + option[i + 1, j + 1] * (1 - p),
                               ST[i, j] - K if is_call else K - ST[i, j]) * np.exp(-r * dt)

    return option[0, 0]


def implied_volatility_binomial_tree(market_price, S, K, T, r, is_call=True):
    sigma = 0.5  # Initial guess
    for i in range(100):
        estimated_price = american_option_binomial_tree(S, K, T, r, sigma, is_call)
        vega = (american_option_binomial_tree(S, K, T, r, sigma + 0.01,
                                              is_call) - estimated_price) / 0.01  # Approximate vega
        sigma -= (estimated_price - market_price) / vega

        if abs(estimated_price - market_price) < 1e-8:  # Convergence check
            return sigma


def RFR():
    bond_data = yf.Ticker("^TNX").history(period="1d")
    return bond_data['Close'].iloc[-1] / 100


def get_options_data(ticker, expiration_date):
    tk = yf.Ticker(ticker)
    if expiration_date in tk.options:
        opt = tk.option_chain(expiration_date)
        return {'calls': opt.calls, 'puts': opt.puts}
    else:
        logging.info(f"No options data for {ticker} on {expiration_date}")
        return None


def tickers_fetch(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM us_quant_filtered")
        return [row[0] for row in cur.fetchall()]


def results_into_db(conn, results):
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO straddle_quant_filtered (ticker, expiration_date, strike_price, call_iv, put_iv) VALUES (%s, %s, %s, %s, %s)",
            results)
    conn.commit()


def main():
    conn = conn_pool.getconn()
    try:
        tickers = tickers_fetch(conn)
        risk_free_rate = RFR()

        for ticker in tickers:
            tk = yf.Ticker(ticker)
            stock_price = tk.info['regularMarketPrice']

            for expiration_date in tk.options:
                options_data = get_options_data(ticker, expiration_date)
                if options_data:
                    for call, put in zip(options_data['calls'].itertuples(), options_data['puts'].itertuples()):
                        if call.strike == put.strike:
                            S = stock_price
                            K = call.strike
                            T = (call.expirationDate - np.datetime64('today')).astype('timedelta64[D]').astype(
                                int) / 365.25

                            call_iv = implied_volatility_binomial_tree(call.lastPrice, S, K, T, risk_free_rate,
                                                                       is_call=True)
                            put_iv = implied_volatility_binomial_tree(put.lastPrice, S, K, T, risk_free_rate,
                                                                      is_call=False)

                            if call_iv < 0.20 and put_iv < 0.20:
                                result = (ticker, expiration_date, K, call_iv, put_iv)
                                results_into_db(conn, [result])

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        conn_pool.putconn(conn)


if __name__ == "__main__":
    main()
