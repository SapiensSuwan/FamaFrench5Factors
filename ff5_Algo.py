import statsmodels.api as sm
from statsmodels import regression
import numpy as np
import pandas as pd
import time
from datetime import date

'''
================================================================================
Before backtest
================================================================================
'''



def initialize(context):
    set_params()
    set_variables()
    set_backtest()


# 1
# set parameteres
def set_params():
    g.tc = 15  # frequency
    g.yb = 63  # days
    g.N = 15  # positions
    g.NoF = 5  # 5 factors


# 2
# set variables
def set_variables():
    g.t = 0  # 记录连续回测天数
    g.rf = 0.04  # risk free rate
    g.if_trade = False  # inday trading or not

    # sort data
    today = date.today()  # get the date of todayxxxx-xx-xx
    a = get_all_trade_days()  # get the trading days:[datetime.date(2005, 1, 4) to datetime.date(2016, 12, 30)]
    g.ATD = [''] * len(a)  # 获得len(a)维的单位向量
    for i in range(0, len(a)):
        g.ATD[i] = a[i].isoformat()  # change the trading date to iso format:2005-01-04到2016-12-30
        # backtest date up to 2016-12-30
        if today <= a[i]:
            break
    g.ATD = g.ATD[:i]


# 3
# set backtest
def set_backtest():
    set_option('use_real_price', True)  # use real price to trade
    log.set_level('order', 'error')
    set_slippage(FixedSlippage(0))


'''
================================================================================
Before trading start
================================================================================
'''


def before_trading_start(context):
    if g.t % g.tc == 0:
        # 每g.tc天，交易一次行
        g.if_trade = True
        # 设置手续费与手续费
        set_slip_fee(context)
        # 设置可行股票池：获得当前开盘的沪深300股票池并剔除当前或者计算样本期间停牌的股票
        g.all_stocks = set_feasible_stocks(get_index_stocks('000300.XSHG'), g.yb, context)
    g.t += 1


# 4 set slippage and transactions costs at different time range
def set_slip_fee(context):
    # set slippage as 0
    set_slippage(FixedSlippage(0))
    # set different transactions costs
    dt = context.current_dt
    log.info(type(context.current_dt))

    if dt > datetime.datetime(2013, 1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    elif dt > datetime.datetime(2011, 1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))

    elif dt > datetime.datetime(2009, 1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))


# 5
# Set up viable stock pool:
# Filter out the stocks that are suspended on the day, and filter out the stocks that have not been suspended in the previous days

def set_feasible_stocks(stock_list, days, context):
    # Get the dataframe, 1 for suspension, 0 for unsuspended
    suspened_info_df = \
    get_price(list(stock_list), start_date=context.current_dt, end_date=context.current_dt, frequency='daily',
              fields='paused')['paused'].T
    # Filter out the stocks that are suspended on the day
    unsuspened_index = suspened_info_df.iloc[:, 0] < 1
    # Get the list of stock symbols that are not suspended on the day:
    unsuspened_stocks = suspened_info_df[unsuspened_index].index
    # Further, filter out the list of stocks that have not been suspended in the previous days:
    feasible_stocks = []
    current_data = get_current_data()
    for stock in unsuspened_stocks:
        if sum(attribute_history(stock, days, unit='1d', fields=('paused'), skip_paused=False))[0] == 0:
            feasible_stocks.append(stock)
    return feasible_stocks


'''
================================================================================
When trading start
================================================================================
'''



def handle_data(context, data):
    if g.if_trade == True:
        # Get date
        todayStr = str(context.current_dt)[0:10]  # Keep Year-Month-Day only
        # Calculate the ai of each stock
        ais = FF(g.all_stocks, getDay(todayStr, -g.yb), getDay(todayStr, -1), g.rf)
        # Allocate money to each stock
        g.everyStock = context.portfolio.portfolio_value / g.N
        # 依打分排序，当前需要持仓的股票
        try:
            stock_sort = ais.sort('score')['code']
        except AttributeError:
            stock_sort = ais.sort_values('score')['code']

        order_stock_sell(context, data, stock_sort)

        order_stock_buy(context, data, stock_sort)

    g.if_trade = False


# 6
# Get a sell signal and execute a sell operation

def order_stock_sell(context, data, stock_sort):
    # For stocks that do not need to be held, sell the whole position
    for stock in context.portfolio.positions:
        # Remove the top g.N stocks
        if stock not in stock_sort[:g.N]:
            stock_sell = stock
            order_target_value(stock_sell, 0)


# 7
# Obtain a buy signal and execute a buy operation

def order_stock_buy(context, data, stock_sort):
    # For stocks that need to be held, buy according to the allocated shares
    for stock in stock_sort:
        stock_buy = stock
        order_target_value(stock_buy, g.everyStock)


# 8
# Calculate the parameters and regress according to the Fama-French 5 factors model, calculate the alpha of the stock and output
def FF(stocks, begin, end, rf):
    LoS = len(stocks)

    q = query(
        valuation.code,
        valuation.market_cap,
        (balance.total_owner_equities / valuation.market_cap / 100000000.0).label("BTM"),
        indicator.roe,
        balance.total_assets.label("Inv")
    ).filter(
        valuation.code.in_(stocks)
    )

    df = get_fundamentals(q, begin)

    # When calculating the 5-factor reinvestment rate, you need to compare it with the data one year ago
    ldf = get_fundamentals(q, getDay(begin, -252))
    # If the data of the previous year does not exist, set Inv=0
    if len(ldf) == 0:
        ldf = df
    df["Inv"] = np.log(df["Inv"] / ldf["Inv"])

    # Select the desired stock portfolio
    try:
        S = df.sort('market_cap')['code'][:LoS / 3]
        B = df.sort('market_cap')['code'][LoS - LoS / 3:]
        L = df.sort('BTM')['code'][:LoS / 3]
        H = df.sort('BTM')['code'][LoS - LoS / 3:]
        W = df.sort('roe')['code'][:LoS / 3]
        R = df.sort('roe')['code'][LoS - LoS / 3:]
        C = df.sort('Inv')['code'][:LoS / 3]
        A = df.sort('Inv')['code'][LoS - LoS / 3:]
    except AttributeError:
        S = df.sort_values('market_cap')['code'][:int(LoS / 3)]
        B = df.sort_values('market_cap')['code'][LoS - int(LoS / 3):]
        L = df.sort_values('BTM')['code'][:int(LoS / 3)]
        H = df.sort_values('BTM')['code'][LoS - int(LoS / 3):]
        W = df.sort_values('roe')['code'][:int(LoS / 3)]
        R = df.sort_values('roe')['code'][LoS - int(LoS / 3):]
        C = df.sort_values('Inv')['code'][:int(LoS / 3)]
        A = df.sort_values('Inv')['code'][LoS - int(LoS / 3):]
    # Get the stock price during the sample period and calculate the daily rate of return
    df2 = get_price(stocks, begin, end, '1d')
    df3 = df2['close'][:]
    df4 = np.diff(np.log(df3), axis=0) + 0 * df3[1:]
    # Calculate the value of the factor
    SMB = sum(df4[S].T) / len(S) - sum(df4[B].T) / len(B)
    HMI = sum(df4[H].T) / len(H) - sum(df4[L].T) / len(L)
    RMW = sum(df4[R].T) / len(R) - sum(df4[W].T) / len(W)
    CMA = sum(df4[C].T) / len(C) - sum(df4[A].T) / len(A)

    # Use CSI 300 as the benchmark for the market
    dp = get_price('000300.XSHG', begin, end, '1d')['close']
    RM = diff(np.log(dp)) - rf / 252

    X = pd.DataFrame({"RM": RM, "SMB": SMB, "HMI": HMI, "RMW": RMW, "CMA": CMA})

    factor_flag = ["RM", "SMB", "HMI", "RMW", "CMA"][:g.NoF]
    print(factor_flag)
    X = X[factor_flag]

    # Apply linear regression on sample data and calculate ai
    t_scores = [0.0] * LoS
    for i in range(LoS):
        t_stock = stocks[i]
        sample = pd.DataFrame()
        t_r = linreg(X, df4[t_stock] - rf / 252, len(factor_flag))
        t_scores[i] = t_r[0]

    # the scores is alpha
    scores = pd.DataFrame({'code': stocks, 'score': t_scores})
    return scores


# 9
# Linear regression function

def linreg(X, Y, columns=3):
    X = sm.add_constant(array(X))
    Y = array(Y)
    if len(Y) > 1:
        results = regression.linear_model.OLS(Y, X).fit()
        return results.params
    else:
        return [float("nan")] * (columns + 1)


# 10
# # Get the date of dt trading days before or after a certain date

def getDay(precent, dt):
    for i in range(0, len(g.ATD)):
        if precent <= g.ATD[i]:
            t_temp = i
            if t_temp + dt >= 0:
                return g.ATD[t_temp + dt]
            else:
                t = datetime.datetime.strptime(g.ATD[0], '%Y-%m-%d') + datetime.timedelta(days=dt)
                t_str = datetime.datetime.strftime(t, '%Y-%m-%d')
                return t_str


'''
================================================================================
After market closed
================================================================================
'''


def after_trading_end(context):
    return
# not needed in this model

