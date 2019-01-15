import pandas as pd
import numpy as np
import datetime
import pickle
from dateutil.relativedelta import relativedelta

""" the script read the data produced by compute_transac_stats.py and for each client with a loan store the
stats for the n months before the loan started """

with open("proc_data/data_transactions_monthlyagregated_all.pkl", "rb") as fopen:
    _dagr = pickle.load(fopen, encoding = "latin1")
with open("proc_data/client_info.pkl", "rb") as fopen:
    _clientinfo = pickle.load(fopen, encoding="latin1")

client_wl_info = _clientinfo[(_clientinfo["has_loan"] == 1)].copy()

window = 60
direction = 1.

if direction > 0:
    _str = "m"
else:
    _str = "mb"

outfile = "proc_data/client_info_extended_{0}{1}.pkl".format(window,_str)
format_str = '%y%m%d'
for _i in range(1,window):
    client_wl_info["balance_avg_{0}mb".format(_i)] = np.NaN
    client_wl_info["balance_max_{0}mb".format(_i)] = np.NaN
    client_wl_info["balance_min_{0}mb".format(_i)] = np.NaN

    client_wl_info["pos_amount_avg_{0}mb".format(_i)] = np.NaN
    client_wl_info["pos_amount_max_{0}mb".format(_i)] = np.NaN
    client_wl_info["pos_amount_min_{0}mb".format(_i)] = np.NaN
    
    client_wl_info["neg_amount_avg_{0}mb".format(_i)] = np.NaN
    client_wl_info["neg_amount_max_{0}mb".format(_i)] = np.NaN
    client_wl_info["neg_amount_min_{0}mb".format(_i)] = np.NaN
    
    client_wl_info["pos_count_{0}mb".format(_i)] = np.NaN
    client_wl_info["neg_count_{0}mb".format(_i)] = np.NaN

_irun = 0
_number_to_proc = len(client_wl_info.account_id.unique())
for _account_id in client_wl_info.account_id.unique():
    if _irun % 50 == 0:
        print ("{0} of {1} client processed".format(_irun, _number_to_proc))
    _irun = _irun + 1
    mask = (client_wl_info.account_id == _account_id)
    dateloan = datetime.datetime.strptime(str(int(client_wl_info[mask]["date_loan"].values[0])), format_str)
    for _i in range(0,window):
        _rangestartloan2 = dateloan + relativedelta(months=direction * _i)
        _rangeendloan2 = dateloan + relativedelta(months=direction * (_i+1))
        if direction < 0:
            mask_dates2 = (_dagr[_account_id].index.values >= np.datetime64(_rangeendloan2)) &\
                (_dagr[_account_id].index.values < np.datetime64(_rangestartloan2))
        else:
            mask_dates2 = (_dagr[_account_id].index.values < np.datetime64(_rangeendloan2)) &\
                (_dagr[_account_id].index.values >= np.datetime64(_rangestartloan2))
        if mask_dates2.sum() == 1:
            client_wl_info.loc[mask,"balance_avg_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["balance"].iloc[0]
            client_wl_info.loc[mask,"balance_min_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["balance_min"].iloc[0]
            client_wl_info.loc[mask,"balance_max_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["balance_max"].iloc[0]

            client_wl_info.loc[mask,"neg_amount_avg_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["neg_amount_avg"].iloc[0]
            client_wl_info.loc[mask,"neg_amount_min_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["neg_amount_min"].iloc[0]
            client_wl_info.loc[mask,"neg_amount_max_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["neg_amount_max"].iloc[0]

            client_wl_info.loc[mask,"pos_amount_avg_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["pos_amount_avg"].iloc[0]
            client_wl_info.loc[mask,"pos_amount_min_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["pos_amount_min"].iloc[0]
            client_wl_info.loc[mask,"pos_amount_max_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["pos_amount_max"].iloc[0]
            
            client_wl_info.loc[mask,"pos_count_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["pos_count"].iloc[0]
            client_wl_info.loc[mask,"neg_count_{0}mb".format(_i+1)] = _dagr[_account_id][mask_dates2]["neg_count"].iloc[0]

        else:
            client_wl_info.loc[mask,"balance_avg_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"balance_min_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"balance_max_{0}mb".format(_i+1)] = np.nan
            
            client_wl_info.loc[mask,"neg_amount_avg_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"neg_amount_min_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"neg_amount_max_{0}mb".format(_i+1)] = np.nan
            
            client_wl_info.loc[mask,"pos_amount_avg_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"pos_amount_min_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"pos_amount_max_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"pos_count_{0}mb".format(_i+1)] = np.nan
            client_wl_info.loc[mask,"neg_count_{0}mb".format(_i+1)] = np.nan


# compute age when loan started
client_wl_info["age"] = (client_wl_info['date2_loan'] - client_wl_info["birthdate"])
client_wl_info["age"] = [e.days/365. for e in client_wl_info["age"]]
# loan payment per month
client_wl_info["permonth"] =client_wl_info["amount"]/client_wl_info["duration"]

#dump info to file
print ("dump {0} client info".format(len(client_wl_info)))
with open(outfile, "wb") as fopen:
    pickle.dump(client_wl_info,fopen)

