import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import pickle
import sys
import matplotlib.pyplot as plt
import ipdb

def map_columns(trans, verbose=0):
    if verbose>5:
        print ("Mapping columns...")

    format_str = '%y%m%d'
    #trans["date2"] = map(lambda x: datetime.datetime.strptime(str(x), format_str), trans["date"])
    trans["date2"] = [datetime.datetime.strptime(str(x), format_str) for x in  trans["date"]]
    trans["sign"] = [1.0 if e == "PRIJEM" else -1.0 for e in trans["type"]]
    trans["amount2"] = trans["amount"] * trans["sign"]
    maptrans_type = {'PRIJEM' :'CREDIT','VYDAJ':'WITHDRAWAL', 'VYBER':"WITHDRAWAL IN CASH"}
    maptrans_operation = {'VKLAD':"CREDIT IN CASH",'PREVOD Z UCTU' : 'COLLECTION FROM OTHER BANK',
                          'VYBER' : "WITHDRAWAL IN CASH", 'PREVOD NA UCET' : "REMITTANCE TO OTHER BANK",
                          'VYBER KARTOU' : "CREDIT CARD WITHDRAWAL", "UNKNOWN" : "UNKNOWN"}
    maptrans_ksymbol = {"POJISTNE": "INSURANCE PAYMENT", "SLUZBY": "PAYMENT FOR STATEMENT",
                        "UROK":"INTEREST CREDITED", "SANKC. UROK":"SANCTION","SIPO":"HOUSEHOLD","DUCHOD":"PENSION",
                        "UVER":"LOAN PAYMENT","UNKNOWN":"UNKNOWN"," ":"UNKNOWN"}
    trans2 = trans.copy()
    trans2["{0}trans".format("type")] = list(map(lambda x: maptrans_type[x], trans["type"] ))
    trans2["operation"].isna().sum()
    trans2["operation"] = trans2["operation"].fillna("UNKNOWN")
    trans2["{0}trans".format("operation")] = list(map(lambda x: maptrans_operation[x], trans2["operation"] ))
    trans2["k_symbol"] = trans2["k_symbol"].fillna("UNKNOWN")
    trans2["{0}trans".format("k_symbol")] = list(map(lambda x: maptrans_ksymbol[x], trans2["k_symbol"] ))

    # READ THE DATE to datatime object
    format_str = '%y%m%d'
    #trans2["date2"] = map(lambda x: datetime.datetime.strptime(str(x), format_str), trans2["date"])
    trans2["date2"] = [datetime.datetime.strptime(str(x), format_str) for x in  trans2["date"]]
    trans2["month"] = [ e.month for e in trans2["date2"]]
    trans2["year"] = [ e.year for e in trans2["date2"]]
    return trans2

def compute_stats(trans2, _id, verbose=5):
    data2 = trans2[trans2["account_id"] == _id].copy()

    if len(data2) == 0:
        print ("NO transactions for acount")
        return 0,0

    # remove duplicated dates and keep the last balance of the date in order to interpolate
    data2_nodup = data2.drop_duplicates(subset="date2", keep='last', inplace=False)
    data2_nodup = data2_nodup.set_index("date2")
    data2_perday = data2_nodup.resample('D').ffill()
    # compute stats per month
    data2_month_avg = data2_perday.resample('M').mean()
    data2_month_min = data2_perday.resample('M').min()
    data2_month_max = data2_perday.resample('M').max()
    aggregate_bal = pd.merge(data2_month_min, data2_month_max, on="date2",suffixes=["_min","_max"])
    aggregate_bal = pd.merge(aggregate_bal, data2_month_avg, on="date2",suffixes=["","_avg"])
    aggregate_bal = aggregate_bal[['balance_min', 'balance_max','balance']]
    aggregate_bal.index.names = ["date"]
    
    # compute stats from positive transactions 
    data_amount_pos = data2[data2["amount2"]>0][["month","year","amount2"]]
    _postrans = data_amount_pos.groupby(["month","year"])
    pos_min = _postrans.min()
    pos_max = _postrans.max()
    pos_avg = _postrans.mean()
    pos_sum = _postrans.sum()
    pos_noper = _postrans.count()
    pos_min["date"] = [datetime.datetime(e[1], e[0], 1)  + relativedelta(day=31) for e in pos_min.index.values] 
    pos_max["date"] = [datetime.datetime(e[1], e[0], 1)  + relativedelta(day=31) for e in pos_max.index.values] 
    pos_avg["date"] = [datetime.datetime(e[1], e[0], 1)  + relativedelta(day=31) for e in pos_avg.index.values]
    pos_sum["date"] = [datetime.datetime(e[1], e[0], 1)  + relativedelta(day=31) for e in pos_sum.index.values]
    pos_noper["date"] = [datetime.datetime(e[1], e[0], 1)  + relativedelta(day=31) for e in pos_noper.index.values]
    

    aggregate_pos = pd.merge(pos_min, pos_max, on="date",suffixes=["_min","_max"])
    aggregate_pos = pd.merge(aggregate_pos, pos_avg, on="date",suffixes=["","_avg"])
    aggregate_pos = pd.merge(aggregate_pos, pos_noper, on="date",suffixes=["_avg","_count"])
    aggregate_pos = pd.merge(aggregate_pos, pos_sum, on="date",suffixes=["_avg","_sum"])
    aggregate_pos=aggregate_pos.set_index("date")
    aggregate_pos.sort_index(inplace=True)
    aggregate_pos.columns =  ['pos_amount_min','pos_amount_max','pos_amount_avg', 'pos_count', 'pos_sum']
    
    # same for negative
    data_amount_neg = data2[data2["amount2"]<0][["month","year","amount2"]]
    _negtrans = data_amount_neg.groupby(["month","year"])
    neg_min = _negtrans.min()
    neg_max = _negtrans.max()
    neg_avg = _negtrans.mean()
    neg_sum = _negtrans.sum()
    neg_noper = _negtrans.count()
    neg_min["date"] = [datetime.datetime(e[1], e[0], 1) + relativedelta(day=31) for e in neg_min.index.values] 
    neg_max["date"] = [datetime.datetime(e[1], e[0], 1) + relativedelta(day=31) for e in neg_max.index.values] 
    neg_avg["date"] = [datetime.datetime(e[1], e[0], 1) + relativedelta(day=31)for e in neg_avg.index.values]
    neg_sum["date"] = [datetime.datetime(e[1], e[0], 1) + relativedelta(day=31)for e in neg_sum.index.values]
    neg_noper["date"] = [datetime.datetime(e[1], e[0], 1) + relativedelta(day=31) for e in neg_noper.index.values]

    aggregate_neg = pd.merge(neg_min, neg_max, on="date",suffixes=["_min","_max"])
    aggregate_neg = pd.merge(aggregate_neg, neg_avg, on="date",suffixes=["","_avg"])
    aggregate_neg = pd.merge(aggregate_neg, neg_noper, on="date",suffixes=["","_count"])
    aggregate_neg = pd.merge(aggregate_neg, neg_sum, on="date",suffixes=["","_sum"])
    aggregate_neg=aggregate_neg.set_index("date")
    aggregate_neg.sort_index(inplace=True)
    aggregate_neg.columns =  ['neg_amount_min','neg_amount_max','neg_amount_avg', 'neg_count', "neg_sum"]
    res = pd.merge(aggregate_pos,aggregate_neg, on="date",how="outer")
    res = pd.merge(res, aggregate_bal, on="date", how="outer")
    if len(aggregate_pos) != len(aggregate_bal) and verbose>5:
        print ("   ",_id,len(aggregate_pos),len(aggregate_bal), len(res))
    res.sort_index(inplace=True)
    
    ## fill nans
    res["neg_sum"] = res["neg_sum"].fillna(0)
    res["pos_sum"] = res["pos_sum"].fillna(0)
    res["neg_count"] = res["neg_count"].fillna(0)
    res["pos_count"] = res["pos_count"].fillna(0)
    res["savings"] = res["pos_sum"]+res["neg_sum"]

    nrep = len(res) * 0.25
    if verbose>10:
        print ("considering nrepeated = {0}".format(nrep))
    repeated = repeated_transactions(data2, ntimes=nrep)
    return res, repeated

def compute_ext_stat(trans2, account_id):
    data2 = trans2[trans2["account_id"] == account_id].copy()
    res,repeated = compute_stats(data2, account_id)
    cols=["amount2","amount","typetrans","operationtrans","k_symboltrans"]
    repeated_trans2=data2[data2[cols].isin(repeated[cols].to_dict('l')).all(1)]
    res_rep, _ = compute_stats(repeated_trans2, account_id)
    if type(res_rep) == type(0) and res_rep == 0:
        res_rep = res.copy()
        res_rep[:] = 0
    else:
        res_rep.columns = [e+"_rep" for e in res_rep.columns]
    results = pd.merge(res, res_rep, on="date",suffixes=["","_rep"], how="left")
    return results

def repeated_transactions(data, ntimes=5, group_on_cols=["account_id", "typetrans","operationtrans","amount","amount2","k_symboltrans"]):
    aux=data.groupby(group_on_cols)
    repeated = []
    for name, group in aux:
        data = list(group[group_on_cols].iloc[0].values)
        repetitions = group.count()["trans_id"]
        data.append(repetitions)
        repeated.append(data)
    out_cols = group_on_cols.copy()
    out_cols.append("repetitions")
    repeated = pd.DataFrame(repeated, columns=out_cols)   
    del out_cols
    repeated = repeated[repeated["repetitions"]>=ntimes]
    return repeated

def plot_averages(trans2, loan, account_id, verbose=0, avg_data=None):

    res,_ = compute_stats(trans2, account_id, verbose=verbose)
    fig,ax = plt.subplots(4,1)
    fig.set_size_inches(18.5, 10.5, forward=True)

    data2 = trans2[trans2["account_id"] == account_id][["trans_id","date","k_symboltrans","date2","balance","amount2","month","year"]]

    ax[0].plot(res.index.values, res["balance"], "-*", color="b",markersize=8, label="avg balance")
    ax[0].fill_between(res.index.values,res["balance_min"] , res["balance_max"], alpha=0.2)
    ax[0].plot(data2["date2"],data2["balance"],".",c="orange", label="balance after transaction")
    if avg_data is not None:
        ax[0].plot(avg_data.index.values, avg_data["balance"], "-", color="b",markersize=8, alpha=0.3)

    ax[0].legend()

    ax[1].plot(res.index.values, res["pos_sum"],"^-",c="blue",label="total amount in positive transactions",)
    ax[1].plot(res.index.values, res["neg_sum"],"^-",c="gray",label="total amount in negative transactions")
    ax[1].plot(res.index.values, res["pos_sum"] + res["neg_sum"],"*-",lw=3,c="green",label="total in month")
    ax[1].axhline(y=0, color="black")
    loantrans = data2[(data2.k_symboltrans=="LOAN PAYMENT")]
    ax[1].plot(loantrans["date2"],loantrans["amount2"],"^",c="red",
               label="amount in loan transaction",markersize=10)
    if avg_data is not None:
        ax[1].plot(avg_data.index.values, avg_data["pos_sum"], "-", color="blue",markersize=8, alpha=0.3)
        ax[1].plot(avg_data.index.values, avg_data["neg_sum"], "-", color="gray",markersize=8, alpha=0.3)
        ax[1].plot(avg_data.index.values, avg_data["pos_sum"] + avg_data["neg_sum"], "-", color="green",markersize=8, alpha=0.3)
    ax[1].legend()


    ax[2].plot(data2["date2"],data2["amount2"],".",c="orange",label="amount in transaction",)
    ax[2].plot(res.index.values, res["pos_amount_avg"],"^",c="blue",label="avg amount in positive transactions",)
    ax[2].fill_between(res.index.values,res["pos_amount_min"], res["pos_amount_max"],alpha=0.2)

    ax[2].plot(res.index.values, res["neg_amount_avg"],"^",c="gray",label="neg_avg amount in negative transactions")
    ax[2].fill_between(res.index.values, res["neg_amount_min"], res["neg_amount_max"],color="gray",alpha=0.2)
    ax[2].plot(loantrans["date2"],loantrans["amount2"],"^",c="red",
               label="amount in loan transaction",markersize=10)


    ax[2].legend()


    ax[3].plot(res.index.values, res["pos_count"],"*-", color="blue", label="number of incoming transactions")
    ax[3].plot(res.index.values, res["neg_count"], "*-", color="gray", label="number of outgoing transactions")
    ax[3].legend(loc="upper left")
    if avg_data is not None:
        ax[3].plot(avg_data.index.values, avg_data["pos_count"], "-", color="blue",markersize=8, alpha=0.3)
        ax[3].plot(avg_data.index.values, avg_data["neg_count"], "-", color="gray",markersize=8, alpha=0.3)

    loangranted = loan[loan["account_id"] == account_id]
    if len(loangranted) == 1:
        format_str = '%y%m%d'
        dateloan = datetime.datetime.strptime(str(loangranted["date"].values[0]), format_str)  
        _dateendloan = dateloan + relativedelta(months=+loangranted.duration.values[0])
        if verbose >=5:
            print ("Account has loan")
            print ("loan started in {0}".format(str(loangranted["date"])))
            print ("loan ends in {0}".format(str(_dateendloan)))
        __text = "status={0} amount={1} duration={2} payments={3}, transactions={4}".format(loangranted.status.values[0],
                loangranted.amount.values[0], 
                loangranted.duration.values[0],loangranted.payments.values[0],len(loantrans))
        hasloan=True
    elif len(loangranted) == 0:
        print ("not loan associated with account")
        dateloan = None
        __text = ""
        hasloan=False
    else:
        print ("More than one loan is associated with account")

    if hasloan:
        if loangranted.status.values[0] in ["A","C"]:
            color = "green"
        else:
            color = "red"
    else:
        color = "black"

    ax[0].text(0.3,0.9,__text,transform=ax[0].transAxes)

    if dateloan is not None:
        ax[0].axvline(x=dateloan)
        ax[1].axvline(x=dateloan)
        ax[2].axvline(x=dateloan)
        ax[0].axvline(x=_dateendloan)
        ax[1].axvline(x=_dateendloan)
        ax[2].axvline(x=_dateendloan)
    
    if hasloan:
        ax[0].set_title("Account ID = {0}, Loan ID = {1}".format(account_id, loangranted.loan_id.values[0]), fontsize=20, color=color)
    else:
        ax[0].set_title("Account ID = {0}".format(account_id), fontsize=20, color=color)

def plot_histograms(summary_dict_real, summary_dict_synth, feature ="balance", ax_stat=["mean", "std"], bins=100, ylim_max=600):
    fig,ax = plt.subplots(1,len(ax_stat),figsize=(15,12))
    
    ax = ax.ravel()

    i = 0
    for stat in ax_stat:
        datab = [summary_dict_real[e].loc[stat,feature] for e in summary_dict_real.keys()]
        datag = [summary_dict_synth[e].loc[stat,feature] for e in summary_dict_synth.keys()]
        _ax=ax[i].hist(datab,color="b",alpha=0.5,bins=bins, label="real data")
        datag = np.array(datag) 
        datag = datag[~(np.isnan(datag))]
        _ax=ax[i].hist(datag,color="r",alpha=0.5,bins=bins, label="synthesized data")
        ax[i].set_xlabel("{0} of monthly {1}".format(stat, feature))
        _lim = ax[i].get_ylim()
        if _lim[1] > ylim_max:
            _lim = (_lim[0],ylim_max)
        ax[i].set_ylim(_lim)
        i = +1
    ax[i].legend()
    fig.suptitle("{0} and {1} of monthly {2}".format(ax_stat[0], ax_stat[1], feature),x=0,y=1.02, fontsize=20, ha="left")
    fig.tight_layout()

def rearange_sum_row(summary_dict, account_id):
    columns = []
    data = []
    for c in (summary_dict[account_id].columns):
        for r in summary_dict[account_id].index:
            c_name = "{0} {1}".format(c,r)
            columns.append(c_name)
            data.append(summary_dict[account_id].loc[r,c])
    return columns, data

def rearange_sum(summary_dict):
    data_all = []
    for k in summary_dict.keys():
        columns,data =   rearange_sum_row(summary_dict, k)
        data_all.append(data)
    return pd.DataFrame(data_all, index =summary_dict.keys(), columns= columns)
    
def compute_avg(avg_data, ids, skip=6):
    columns = avg_data[ids[0]].columns
    sum_data = avg_data[ids[0]][skip:]
    sum_data["data"] = 1
    for _id in ids[1:]:
        avg_data[_id]["data"] = 1
        sum_data = sum_data.add(avg_data[_id][skip:], fill_value=0)
    for _c in columns:
        sum_data[_c] = sum_data[_c] / sum_data["data"]

    std_data = avg_data[ids[0]][skip:].add(-1 * sum_data)**2
    std_data["data"] = 1
    for _id in ids[1:]:
        _aux = avg_data[_id][skip:][columns].add(-1 * sum_data)**2
        std_data = std_data.add(_aux, fill_value=0)
    for _c in columns:
        std_data[_c] = np.sqrt(std_data[_c] / sum_data["data"])

    return sum_data, std_data




if __name__ == "__main__":


    ############################
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    trans = pd.read_csv(input_file,sep=";")

    trans2 = map_columns(trans)

    print ("Generating features")

    _irun=0
    print ("Going through {0} accounts".format(len(trans2.account_id.unique())))
    _acc_id = trans2.account_id.unique()
    results = []

    summary = False # set to True to store summary instead of monthly averages
    
    _irun = 0
    for _id in _acc_id:
        __results = compute_ext_stat(trans2, _id)
        if summary:
            results.append(__results.describe())
        else:
            results.append(__results)
        _irun +=1
        if _irun % 100 == 0:
            print ("processed {0} of {1} accounts".format(_irun, len(_acc_id)))

    dict_balances = dict(zip(_acc_id, results))

    with  open(output_file, "wb") as fopen:
        pickle.dump(dict_balances, fopen)   
