
# coding: utf-8

# In[668]:


#testing range of dates
start_date = '2017-07-23'
end_date = '2018-07-23'

#Portfolio size
portfolio_size = 10


# In[340]:


#grab closing data from s&p as a whole
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key = 'BEII34ZY6UI335CP', output_format = 'pandas')
indicies_av, metadata = ts.get_daily_adjusted('^GSPC', outputsize = 'full')
s_and_p_data = indicies_av.loc[five_yrs_ago:]["4. close"]


# In[147]:


import numpy as np
#pandas 0.22.0 required at the moment for Stocker: use pip install pandas=0.22.0
import pkg_resources
pkg_resources.require("pandas==0.22.0")
import pandas as pd
from pandas_datareader import data as wb
from datetime import datetime
from dateutil.relativedelta import relativedelta
from stocker import Stocker
import matplotlib.pyplot as plt
import math
from alpha_vantage.timeseries import TimeSeries

#Set up time series with key
ts = TimeSeries(key = 'BEII34ZY6UI335CP', output_format = 'pandas')

#Get 5 years ago from today because IEX only goes five years into the past
five_yrs_ago = str(datetime.now() - relativedelta(years=5))
#print(five_yrs_ago)

#read the stocks in S&p as a csv file from this github link to get tickers
s_and_p = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")
#s_and_p = ['ABT', 'ABBV', 'ACN', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']

#create a pandas DataFrame for the data_set
data_set = pd.DataFrame()

#loop through each ticker of the S&p and get 5years of closing data from iex and add to data frame
for t in s_and_p['Symbol']:
    data_set[t] = wb.DataReader(t, data_source = 'iex', start = five_yrs_ago)["close"]

#Remove any stocks that are no longer public (have NaN as last value)
for t in data_set:
    if(math.isnan(data_set[t].tail(1))):
        del data_set[t]

#grab closing data from s&p as a whole from alpha vantage
indicies_av, metadata = ts.get_daily_adjusted('^GSPC', outputsize = 'full')
s_and_p_data = indicies_av.loc[five_yrs_ago:]["4. close"]


# In[260]:


#to save and display s&p data (not necessary)
data_set
#data_set.to_csv('s&p 500 stocks shortened.csv')


# In[645]:


#calculate daily returns for all the stocks and s&p as a whole
returns = (data_set.loc[start_date:end_date] / data_set.loc[start_date:end_date].shift(1)) - 1
s_and_p_returns = (s_and_p_data.loc[start_date:end_date] / s_and_p_data.loc[start_date:end_date].shift(1)) - 1 

#print(returns)

#calculate annual returns
stocks_annual = returns.mean() * 250
s_and_p_annual = s_and_p_returns.mean() * 250

#grab the top (portfolio_size) stocks from the s&p and add them to a series
top_portfolio_returns = stocks_annual.sort_values().tail(portfolio_size)

#print(top_portfolio_returns)

#Tickers
portfolio_tickers = []
for key, returns in top_portfolio_returns.to_dict().items():
    portfolio_tickers.append(key)

#print(portfolio_tickers)

#make a np array of size portfolio_size and initiate it with random values
weights = np.random.random(portfolio_size)

#get the most recent closing value for each stock and add it to the weights array
i = 0 #keep track of current index
for t in portfolio_tickers:
   # print(data_set[t].tail(1).iloc[0])
    weights[i] = data_set[t].tail(1).iloc[0]
    i += 1
    
#print(weights)

#divide each value in weights by the sum of value in weights to create a weighting by closing price for our portfolio
weights /= np.sum(weights)

#type(weights)
    
#print(weights)


# In[646]:


#Get daily returns for all stocks in our portfolio and place them in portfolio_returns
returns = (data_set.loc[start_date:end_date] / data_set.loc[start_date:end_date].shift(1)) - 1
portfolio_returns = pd.DataFrame()
for t in portfolio_tickers:
    portfolio_returns[t] = returns[t]


# In[647]:


def clean(x):
    return str(round(x, 5) * 100) + "% "


# In[648]:


print('Top ' + str(portfolio_size) + ' Returns in S&P (by %):')
print(clean(top_portfolio_returns))
print()
print('Annual return of S&P as a whole:')
print(clean(s_and_p_annual))
print()
print('Annual Portfolio Return:')
print(clean(np.dot(top_portfolio_returns, weights))) #Calculate the dot product of the top annual returns and the weights


# In[674]:


portfolio_var = np.dot(weights.T, np.dot(portfolio_returns.cov() * 250, weights))
print('Portfolio Variablility:')
print(clean(portfolio_var))


# In[675]:


portfolio_vol = (np.dot(weights.T, np.dot(portfolio_returns.cov() * 250, weights))) ** 0.5
print('Portfolio Volatility')
print(clean(portfolio_vol))


# #make an array for the data we will get for the portfolio
# portfolio_data = []
# 
# #loop through stocks in portfolio to get their data as a Stocker object
# for t in portfolio_tickers:
#     portfolio_data.append(Stocker(t))
#     
# portfolio_data

# In[676]:


portfolio_prices = pd.DataFrame()
for t in portfolio_tickers:
    portfolio_prices[t] = data_set[t]

(portfolio_prices / portfolio_prices.iloc[0] * 100).plot(figsize=(10,5))
plt.show()


# In[618]:


pd.set_option('use_inf_as_null', True)
variables = pd.read_csv('multivariable.csv')
variables.set_index('Dates', inplace=True)
variables

margins=pd.read_csv("profit margins.csv")
margins.set_index('Dates', inplace=True)


# In[677]:


variables


# In[669]:


from scipy import stats
import statsmodels.api as sm

start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y").lstrip("0")
end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y").lstrip("0")
start_date


# In[670]:


MKT_CAP = []
VOLUME = []
REVENUE = []
EST_EPS_GAAP = []
NET_DEBT_TO_EBITDA = []
VOLATILITY_10D = []
VOLATILITY_30D = []
VOLATILITY_90D = []
MARGINS = []
MULTIVARIABLE = []
#ARRAY_2D = [][]
row_length = len(variables.iloc[0])

i = 0
while i < row_length - 9:
    Y = variables[str(i)].loc[start_date:end_date].astype(float)
    X = pd.DataFrame([variables[str(i+4)].loc[start_date:end_date], variables[str(i+2)].loc[start_date:end_date], variables[str(i+3)].loc[start_date:end_date], variables[str(i+5)].loc[start_date:end_date], variables[str(i+8)].loc[start_date:end_date], margins.iloc[:,i % 9].loc[start_date:end_date]]).transpose()
    X = X.astype(float)
    X1 = sm.add_constant(X)
    try:
        reg = sm.OLS(Y.astype(float), X1.astype(float), "drop").fit()
    except ValueError:
        "Do Nothing"
    else:
        MULTIVARIABLE.append(reg.rsquared)
    for j in range(1,10):
        if j == 9:
            X = margins.iloc[:,i % 9].loc[start_date:end_date].astype(float)
        else:
            X = variables[str(i+j)].loc[start_date:end_date].astype(float)
        slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
        #ARRAY_2D[i % 9][ j - 1].append([r_value, p_value])
        if j == 1:
            MKT_CAP.append([r_value, p_value])
        elif j == 2:
            VOLUME.append([r_value, p_value])
        elif j == 3:
            REVENUE.append([r_value, p_value])
        elif j == 4:
            EST_EPS_GAAP.append([r_value, p_value])
        elif j == 5:
            NET_DEBT_TO_EBITDA.append([r_value, p_value])
        elif j == 6:
            VOLATILITY_10D.append([r_value, p_value])
        elif j == 7:
            VOLATILITY_30D.append([r_value, p_value])
        elif j == 8:
            VOLATILITY_90D.append([r_value, p_value])
        elif j == 9:
            MARGINS.append([r_value, p_value])
    #Potentiall to keep data for multivariable here
    i += 9
    
ARRAY_2D = MKT_CAP, VOLUME, REVENUE, EST_EPS_GAAP, NET_DEBT_TO_EBITDA, VOLATILITY_10D, VOLATILITY_30D, VOLATILITY_90D, MARGINS


# In[671]:


def top_momentum_reg():
    MKT_CAP = []
    VOLUME = []
    REVENUE = []
    EST_EPS_GAAP = []
    NET_DEBT_TO_EBITDA = []
    VOLATILITY_10D = []
    VOLATILITY_30D = []
    VOLATILITY_90D = []
    MARGINS = []
    MULTIVARIABLE = []
    #ARRAY_2D = [][]
    row_length = len(variables.iloc[0])

    i = 0
    while i < row_length - 9:
        try:
            #See if the current ticker is one the top momentum tickers
            portfolio_tickers.index(variables[str(i)].iloc[0].split()[0])
        except ValueError:
            "Do Nothing"
        else:
            Y = variables[str(i)].loc[start_date:end_date].astype(float)
            Y = variables[str(i)].loc[start_date:end_date].astype(float)
            X = pd.DataFrame([variables[str(i+4)].loc[start_date:end_date], variables[str(i+2)].loc[start_date:end_date], variables[str(i+3)].loc[start_date:end_date], variables[str(i+5)].loc[start_date:end_date], variables[str(i+8)].loc[start_date:end_date], margins.iloc[:,i % 9].loc[start_date:end_date]]).transpose()
            X = X.astype(float)
            X1 = sm.add_constant(X)
            try:
                reg = sm.OLS(Y.astype(float), X1.astype(float), "drop").fit()
            except ValueError:
                "Do Nothing"
            else:
                MULTIVARIABLE.append(reg.rsquared)
            for j in range(1,10):
                if j == 9:
                    X = margins.iloc[:,i % 9].loc[start_date:end_date].astype(float)
                else:
                    X = variables[str(i+j)].loc[start_date:end_date].astype(float)
                slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
                #ARRAY_2D[i % 9][ j - 1].append([r_value, p_value])
                if j == 1:
                    MKT_CAP.append([r_value, p_value])
                elif j == 2:
                    VOLUME.append([r_value, p_value])
                elif j == 3:
                    REVENUE.append([r_value, p_value])
                elif j == 4:
                    EST_EPS_GAAP.append([r_value, p_value])
                elif j == 5:
                    NET_DEBT_TO_EBITDA.append([r_value, p_value])
                elif j == 6:
                    VOLATILITY_10D.append([r_value, p_value])
                elif j == 7:
                    VOLATILITY_30D.append([r_value, p_value])
                elif j == 8:
                    VOLATILITY_90D.append([r_value, p_value])
                elif j == 9:
                    MARGINS.append([r_value, p_value])
        #Potentiall to keep data for multivariable here
        i += 9
    
    ARRAY_2D = [MKT_CAP, VOLUME, REVENUE, EST_EPS_GAAP, NET_DEBT_TO_EBITDA, VOLATILITY_10D, VOLATILITY_30D, VOLATILITY_90D, MARGINS, MULTIVARIABLE]
    return ARRAY_2D


# In[672]:


def var_name(arg):
    names = {
        1 : 'Market Cap',
        2 : 'Volume',
        3 : 'Revenue',
        4 : 'Estimated Earnings Per Sale GAAP',
        5 : 'Net Debt to EBIDTA',
        6 : 'Volatility 10 Days',
        7 : 'Volatility 30 Days',
        8 : 'Volatility 90 Days',
        9 : 'Profit Margins'
    }
    return names.get(arg)


# In[673]:


ARRAY_2D = MKT_CAP, VOLUME, REVENUE, EST_EPS_GAAP, NET_DEBT_TO_EBITDA, VOLATILITY_10D, VOLATILITY_30D, VOLATILITY_90D, MARGINS, MULTIVARIABLE
R = []
P = []
print('Total S&P 500 Data')
print()
for i in range(1,10):
    for t in ARRAY_2D[i-1]:
        R.append(t[0])
        P.append(t[1])
    print(var_name(i), "R Squared Value", round(np.nanmean(R) ** 2, 5))
    print(var_name(i), "P Value", round(np.nanmean(P), 5))
    print()
print("Multivariable R Squared Value", round(np.nanmean(ARRAY_2D[9]) ** 2, 5))    
print()
print()

ARRAY_2D = top_momentum_reg()

print('Top ' + str(portfolio_size) + ' Momentum Stocks in S&P Data')
print()
R = []
P = []
for i in range(1,10):
    for t in ARRAY_2D[i-1]:
        R.append(t[0])
        P.append(t[1])
    print(var_name(i), "R Squared Value", round(np.nanmean(R) ** 2, 5))
    print(var_name(i), "P Value", round(np.nanmean(P), 5))
    print()
print("Multivariable R Squared Value", round(np.nanmean(ARRAY_2D[9]) ** 2, 5))


# In[468]:


from scipy import stats
import statsmodels.api as sm

row_length = len(variables.iloc[0])

#Y = (Y.astype(float) / Y.shift(1).astype(float)) - 1
#i = 1269
Y = variables[str(i)].iloc[1:]
Y = Y.astype(float)
#print(Y)

X = variables[str(i+2)].iloc[1:]
#X = pd.DataFrame([variables[str(i+4)].iloc[1:], variables[str(i+2)].iloc[1:], variables[str(i+3)].iloc[1:], variables[str(i+5)].iloc[1:], variables[str(i+8)].iloc[1:]]).transpose()
X = X.astype(float)
#X = [list(map(float, variables['1'].tolist()[1:])), list(map(float, variables['2'].tolist()[1:]))]
#print(X)
X1 = sm.add_constant(X)

reg = sm.OLS(Y.astype(float), X1.astype(float), "drop").fit()

reg.summary()

 X = pd.DataFrame([variables[str(i+4)].iloc[1:], variables[str(i+2)].iloc[1:], variables[str(i+3)].iloc[1:], variables[str(i+5)].iloc[1:], variables[str(i+8)].iloc[1:]]).transpose()
            X = X.astype(float)
            X1 = sm.add_constant(X)
            reg = sm.OLS(Y.astype(float), X1.astype(float), "drop").fit()
            MULTIVARIABLE[i%9] = reg.rsquared

#slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
#print(r_value)
#print(r_value ** 2)
#print(X)
#MKT_CAP = []
#VOLUME = []
#REVENUE = []
#EST_EPS_GAAP = []

#i = 1
#while i < row_length:
#    for j in 
#    i += 9
    


# In[469]:


slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
print(r_value)
print(r_value ** 2)

