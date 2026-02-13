#staring with Monte Carlo Simulation for the Stovks 
#this will estimate the  1-day 99% VALUE AT RISK (VaR).

#Step 1 Portfolio Configuration 

import json 
Port_config = {
    "tickers": ["AAPL", "MSFT", "NVDA", "GOOG", "AMD"],
    "weights": [0.3, 0.25, 0.2, 0.15, 0.1], 
    "start": "2018-01-01",
    "end": "2026-01-31",
    "train_start": "2018-01-01",
    "train_end": "2024-12-31",
    "test_start": "2024-12-31",
    "test_end": "2025-09-30",
    "confidence": 0.99
}

with open("portfolio_config.json", "w") as f:
    json.dump(Port_config, f, indent=9)

#step 2 downloading the data and calculating the returns 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf 
from scipy.stats import norm 
from IPython.display import Markdown


def down_price_data(tickers, start_date="2018-01-01",end_data="2026-01-31"):
    data= yf.download(tickers, start= start_date, end=end_data, auto_adjust=True, progress=False)['Close']
    return data

with open("portfolio_config.json", "r") as f:
    Port_config = json.load(f)

tickers = Port_config["tickers"]
weights = Port_config["weights"]
start = Port_config["start"]
end = Port_config["end"]
train_start = Port_config["train_start"]
train_end = Port_config["train_end"]
test_start = Port_config["test_start"]
test_end = Port_config["test_end"]
confidence = Port_config["confidence"]

# we will use T_T_P as the price from start to end date, which will be used for the rolling backtest

T_T_P = down_price_data(tickers, start, end)

Portfo1_T_P = down_price_data(tickers, train_start, train_end)
Portfo1_T_P.head()

#Calculating the daily return based on individual assets

def Portfo_returns(price_data, weights= None, normalize = True):

    try:
        if weights is None:
            weights = np.array([1/price_data.shape[1]]*price_data.shape[1])
        else:
            weights = np.array(weights)
        if len(weights) != price_data.shape[1]:
            raise ValueError('The asset and the weight do not match')
        
        if normalize:
            weights = weights/np.sum(weights)
        
        log_returns = np.log(price_data/price_data.shift(1)).dropna()

        Portfo_returns = log_returns.dot(weights)                    #Coumputing log returns as weighted sum of individual asset log returnss

        return Portfo_returns
    
    except ValueError as e:
        print(f"Cannot Calculate:{e}")
        return None
    
    except Exception as e:
        print(f"Cannot Calculate:{e}")
        return None
    
Portfo1_Total_P = Portfo_returns(Portfo1_T_P, weights)

Total_R = Portfo_returns(T_T_P, weights)

print("Traning Portfolio Returns (First 10 rows):")
print(Portfo1_Total_P.head())
print("_________________________________________________________________________________________")

print("\nFull Period Portfolio Return (First 10 rows):")
print(Total_R.head())
print("_________________________________________________________________________________________")

#step 3 Monte Carlo Simulation

def MtCr_VaR(Portfo_returns, confidence=0.99, simulations=100000, return_path=True):

    try:
        mean = Portfo_returns.mean()
        std = Portfo_returns.std()

        simulated = np.random.normal(mean, std, size=simulations)

        VaR = np.percentile(simulated, (1 - confidence)*100)

        if return_path:
            return VaR, simulated
        else:
            return VaR
        
    except Exception as e:
        print(f"Issue:{e}")
        return None
    
VaR_train, sim_train = MtCr_VaR(Portfo1_Total_P, confidence=0.99)
print(f"\n1-day 99% Monte Carlo VaR (Training Period): {VaR_train:.5f}")
print("_________________________________________________________________________________________")

VaR_full, sim_full = MtCr_VaR(Total_R, confidence=0.99)
print(f"1-day 99% Monte Carlo VaR (Full Period): {VaR_full:.5f}")
print("_________________________________________________________________________________________")

print("Applying Function: MtCr_VaR()")
print(f"Simulating 1 day VaR using:")
print(f"- Training period: {train_start} to {train_end}")
print(f"- Confidence Level: {confidence*100:.0f}%")
print("_________________________________________________________________________________________")

Portfo1_MtCr_VaR, Portfo1_MtCr_VaR_Path  = MtCr_VaR(Portfo1_Total_P)

print("1 - Day VaR Estimation")
print(f"Estimated 99% 1-day VaR: {abs(Portfo1_MtCr_VaR):.2%}")
print(f"There is a 1% chance that the portfolio may loose more tham {abs(Portfo1_MtCr_VaR):.2%} in one day under normal market conditions.")
print("_________________________________________________________________________________________")

def plot_mc_distribution(simulated_returns, var, confidence=0.99, absolute_var=True, show_normal=True):
    plt.figure(figsize=(10,5))

    plt.hist(simulated_returns, bins=100, alpha = 0.7, color='skyblue', edgecolor = 'k', density=True)

    if show_normal:
        mu = np.mean(simulated_returns)
        sigma = np.std(simulated_returns)
        x = np.linspace(min(simulated_returns), max(simulated_returns), 1000)
        plt.plot(x, norm.pdf(x, mu, sigma), color ='black', linestyle='-', linewidth = 2, label='Normal Fit')

    label_var = -var if absolute_var else var 
    plt.axvline(var, color='red', linestyle='--', linewidth=2, 
                label=f'{int(confidence*100)}% VaR = {label_var:.2%}')
    
    plt.title("Monte Carlo Simulation Portfolio Return Distribution")
    plt.xlabel("Simulation Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_mc_distribution(sim_train, VaR_train)
plot_mc_distribution(sim_full, VaR_full)

plot_mc_distribution(Portfo1_MtCr_VaR_Path, Portfo1_MtCr_VaR, confidence=confidence)

print("Monte Carlo Simulation Summary")
print(f"The Histogram shows all 1-day simulated returns.")
print(f"The red dashed Line marks the {int(confidence*100)}% VaR at {abs(Portfo1_MtCr_VaR):.2%}")
print(f"Only {int((1 - confidence)*100)}% of simulated paths fall below this threshold")
print("_________________________________________________________________________________________")

#step 4 BACKTESTING

test_start = Port_config["test_start"]
test_end = Port_config["test_end"]
test_prices = down_price_data(tickers, test_start, test_end)
test_returns = Portfo_returns(test_prices, Port_config["weights"])

#Comparing actual var with simulated var
# counting breach that is how many times does the model  make a loss in actual market CCORDING TO THE BASELS REQUIREMNET 
#Backtesting using the Basel Traffic light framework 
#Green Zone (0–4 breaches)
#Your model is good.
#Bank uses normal capital requirement.

# Yellow Zone (5–9 breaches)
#Your model is questionable.
#Bank must multiply capital by a penalty factor.

# Red Zone (10+ breaches)
#Your model is bad.
#Bank must hold much more capital.
#This is called the Basel Traffic Light Backtesting Framework.

#Why does Basel care?
#(Because banks use VaR to decide how much money they must keep aside to survive bad market days.
#If your VaR model is too optimistic, the bank might not have enough capital during a crisis.
#So Basel forces banks to:
#- backtest their VaR
#- count breaches
#- increase capital if the model performs poorly)

def backtest_VaR(real_returns, VaR):
    violations = real_returns < VaR
    num_violations = violations.sum()
    total_days = len(real_returns)
    violations_rate = num_violations/total_days

    if num_violations <= 4:
        zone = "GREEN - Safe"
    elif num_violations <=9:
        zone = "YELLOW - Possible risk "
    else:
        zone = "RED - Not Safe"

    return{
        "violations": int(num_violations),
        "total": total_days,
        "rate": round(violations_rate, 4),
        "zone": zone
    }

backtest_VaR(test_returns, Portfo1_MtCr_VaR)

#Plotting backtesting results 

x = test_returns.index
y = test_returns.values.astype("float64")
y2 = np.full_like(y, Portfo1_MtCr_VaR)
mask = y < Portfo1_MtCr_VaR

plt.figure(figsize=(12, 5))
plt.plot(x, y, label = "Actual Portfolio Return", color = "blue", linewidth = 1)
plt.axhline(y = Portfo1_MtCr_VaR, color = "red", linestyle = "--", linewidth = 2, label = f"99% VaR = {Portfo1_MtCr_VaR:.2%}")
plt.fill_between(x, y, y2, where=mask, color = "red", alpha = 0.3, label = "violations")

plt.title("Backtesting: Actual Returns VS 99% VaR")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

result = backtest_VaR(test_returns, Portfo1_MtCr_VaR)

print(f"Total Days Tested: {result['total']}")
print(f"Number of VaR Breaches: {result['violations']}")
print(f"Breach Rate: {result['rate']*100:.2f}%")
print(f"Basel Zone: {result['zone']}")
print("_________________________________________________________________________________________")

#step 5  Further analysis - using Rolling Monte Carlo Backtest

#A rolling 250‑day window recalculates VaR each day using the latest data, 
# making the model responsive to changing market volatility. 
# Breaches are then tracked daily to assess how well the VaR model performs over time.

def rolling_mc_var_bk_tst(returns, test_start, test_end=None, window=250, num_sim = 100000, confidence = 0.99):

    test_start = pd.to_datetime(test_start)
    if test_start not in returns.index:
        raise ValueError(f"test_start{test_start} not in return index.")
    
    start_idx = returns.index.get_loc(test_start)        #row position of test_ start

    #considering the historical data for further calculations

    if start_idx <  window:
        raise ValueError(f"Not enough data test_start to construct{window}-day window")

    #determining test_end 
    if test_end is not None:
        test_end = pd.to_datetime(test_end)
        if test_end not in returns.index:
            raise ValueError(f"test_end{test_end} not in retrun index.")
        end_idx = returns.index.get_loc(test_end)

    else:
        end_idx = len(returns) - 1           # that is using all remaing data

    #Rolling backtest loop

    rolling_var = []
    real_returns = []
    dates = []

    for i in range(start_idx, end_idx):
        train = returns[i - window:i]            #historical data i.e previous 250 days 
        mu = train.mean()
        sigma = train.std()

        simulated = np.random.normal(mu, sigma, num_sim)
        var = np.percentile(simulated, (1 - confidence)*100)

        rolling_var.append(var)
        real_returns.append(returns.iloc[i + 1])
        dates.append(returns.index[i + 1])

    result_df =  pd.DataFrame({
        'Date': dates,
        'Real_Return': real_returns,
        'Rolling_VaR': rolling_var
    }).set_index('Date')

    result_df['Violation'] = result_df['Real_Return'] < result_df['Rolling_VaR']

    return result_df

result_df = rolling_mc_var_bk_tst(Total_R, test_start, test_end)

total_violations = result_df['Violation'].sum()
total_days = len(result_df)
violation_rate = total_violations/total_days

print("Rolling VaR Backtest Summary")
print(f"Total Days Tested: {total_days}")
print(f"Total Violations: {total_violations}")
print(f"Violation Rate: {violation_rate:.2%}")

if total_violations <= 4:
    zone = "GREEN - Safe"

elif total_violations <= 9:
    zone = "YELLOW - Possible Risk"

else:
    zone = "RED - Not Safe"

print(f"Basel Zone: {zone}")
print("_________________________________________________________________________________________")

rolling_result = rolling_mc_var_bk_tst(
    returns=Total_R, 
    test_start =  "2024-12-31",
    test_end =  "2025-09-30"
)


x = rolling_result.index
y = rolling_result['Real_Return'].values.astype("float64")
y2 = rolling_result['Rolling_VaR'].values.astype("float64")
mask = y < y2

plt.figure(figsize=(12, 5))
plt.plot(x, y, label = "Actual Portfolio Return", color = "blue", linewidth = 1)
plt.plot(x, y2, color = "red", linestyle = "--", label = "99% Rolling VaR")
plt.fill_between(x, y, y2, where=mask, color = "red", alpha = 0.3, label = "Violations")

plt.title("Rolling Backtest: Actual Returns vs 99% VaR")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

total_days = len(rolling_result)
violations = rolling_result['Violation'].sum()
violation_rate = violations/total_days

print(f"Test Period: {total_days} days")
print(f"Violations: {violations}")
print(f"Violation rate: {violation_rate:.2%}")

if total_violations <= 4:
    zone = "GREEN - Safe"

elif total_violations <= 9:
    zone = "YELLOW - Possible Risk"

else:
    zone = "RED - Not Safe"

print(f"Basel Zone: {zone}")
print("Based on the number of violations, the model falls into the {zone} zone under Baasel III.")
print("_________________________________________________________________________________________")


