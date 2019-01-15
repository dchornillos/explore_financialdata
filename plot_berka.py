import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.model_selection import learning_curve

def plot_todateloan(dagr, loan, account_id, color="b",alpha=0.3,ax=None, keytoplot="balance"):
    if ax is None:
        fig,ax = plt.subplots(1,1)

    res = dagr[account_id]
    res.sort_index(inplace=True)

    __loangranted = loan[loan["account_id"] == account_id]
    if len(__loangranted) == 1:
        format_str = '%y%m%d'
        dateloan = datetime.datetime.strptime(str(__loangranted["date"].values[0]), format_str)  
    elif len(__loangranted) == 0:
        print ("not loan associated with account")
        dateloan = None
    else:
        print ("More than one loan is associated with account")

    ax.plot([np.timedelta64(e, 'D')/30. for e in (res.index.values - np.datetime64(dateloan))], res[keytoplot],
            "-", color=color, label ="", markersize=8,alpha=alpha,lw=1)


def plot_fromloan(data, window=6, alpha=0.3,ax=None, param="balance_avg", 
                 direction=-1, put_legend=True, color1="b", color2="r", plotminmax=True):
    if ax is None:
        fig,ax = plt.subplots(1,1)
        fig.set_size_inches(18.5, 10.5, forward=True)

    maskp = (data["status2"] == 0)
    maskn = (data["status2"] == 1)

    x = []
    ymean = []
    ystd = []
    ymin = []
    ymax = []
    ymean_n = []
    ystd_n = []
    ymin_n = []
    ymax_n = []
    for i in range(1,window):
        y1 = data[maskp]["{1}_{0}mb".format(i, param)]
        ymean.append(y1.mean())
        ystd.append(y1.std())
        ymin.append(y1.max())
        ymax.append(y1.min())
        x.append(direction*i)
        y2 = data[maskn]["{1}_{0}mb".format(i, param)]
        ymean_n.append(y2.mean())
        ystd_n.append(y2.std())
        ymin_n.append(y2.max())
        ymax_n.append(y2.min())
    ymean = np.array(ymean)
    ystd = np.array(ystd)
    ymean_n = np.array(ymean_n)
    ystd_n = np.array(ystd_n)
    ax.plot(x,ymean,"-",color=color1,alpha=0.7,lw=3, label="loans OK")
    if plotminmax:
        ax.plot(x,ymin,"--",color=color1,alpha=0.3,lw=1)
        ax.plot(x,ymax,"--",color=color1,alpha=0.3,lw=1)
    ax.fill_between(x, ymean-ystd, ymean+ystd, alpha=0.3,color=color1) 
    #ax.fill_between(x, ymin, ymax, alpha=0.06,color=color1) 
    ax.plot(x,ymean_n,"-",color=color2,lw=3,alpha=0.7, label="client in debt")
    if plotminmax:
        ax.plot(x,ymin_n,"--",color=color2,alpha=0.3,lw=1)
        ax.plot(x,ymax_n,"--",color=color2,alpha=0.3,lw=1)
    ax.fill_between(x, ymean_n-ystd_n, ymean_n+ystd_n,color=color2, alpha=0.3)  
    #ax.fill_between(x, ymin_n, ymax_n, alpha=0.06,color=color2) 
    #ax.set_xlabel("months from loan start date", fontsize=16)
    #ax.set_ylabel("average balance")
    if put_legend:
        ax.legend(loc="upper left", fontsize=16)

# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5),color1 = "g",color2 = "r",
                        plotmedian=True, includebasetext=False, baselinepos=[180,0.9]):
    """Generate a simple plot of the test and training learning curve"""
    fig,ax = plt.subplots(1,1,figsize=(10,7))
    plt.title(title, color="dimgray", loc="left", size=20)
    plt.xlabel("Training sample size", color="dimgrey",size=15)
    plt.ylabel("Accuracy score", color="dimgrey",size=15)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print ("number of element in train sample = ", train_sizes)

    
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_median = np.median(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    if plotmedian:
        test_scores_median = np.median(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #plt.grid()
    plt.plot(train_sizes[:], np.zeros(len(train_sizes))+0.889, ls="--", color = "dimgrey", lw=2)
    if includebasetext:
        plt.text(baselinepos[0], baselinepos[1], "baseline prediction",color="dimgrey",size=15)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,
                     color=color2)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color=color1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color=color2, lw=2,
             label="Training")
    if plotmedian:
        plt.plot(train_sizes, train_scores_median, 'x:', color=color2,alpha=0.3,
             label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=color1,
             label="Cross-validation", lw=3)
    if plotmedian:
        plt.plot(train_sizes, test_scores_median, 'x:', color=color1, alpha=0.3,
             label="Cross-validation")
    plt.ylim([0.8,1.05])

    legend=plt.legend(loc="best")
    plt.setp(legend.get_texts(), color='dimgray', size=15)

    ax.spines['bottom'].set_color('dimgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('dimgray')
    ax.tick_params(axis='x', colors='dimgray')
    ax.tick_params(axis='y', colors='dimgray', size=3)
    ax.get_yaxis().set_ticks([0.8,0.9,1.0])

    ax.tick_params(labelsize=14)

    return plt
