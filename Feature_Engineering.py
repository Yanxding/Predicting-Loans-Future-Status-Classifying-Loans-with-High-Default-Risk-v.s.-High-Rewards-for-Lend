'''
Project: Predicting Loan Status
Script: Feature Engineering
Auther: Yanxiang Ding
Date: March 2019

Data used: All Lending Club loan data
    (https://www.kaggle.com/wordsforthewise/lending-club)
'''


# ========== import packages ==========
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib qt5


# ========== import cleaned data ==========
x_all = pd.read_csv('data_exy_new.csv')
y = pd.read_csv('data_y_newnew.csv')


# ========== select and standardize non-categorical features ==========
no_category_x = ['loan_amnt','funded_amnt','funded_amnt_inv','int_rate','installment','annual_inc','dti','delinq_2yrs','fico_range_low','inq_last_6mths',
                 'mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','out_prncp','out_prncp_inv',
                 'total_pymnt', 'total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee',
                 'last_pymnt_amnt','last_fico_range_high','last_fico_range_low','collections_12_mths_ex_med','tot_coll_amt','tot_cur_bal','max_bal_bc',
                 'total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m','avg_cur_bal','bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt',
                 'mort_acc','mths_since_recent_bc_dlq','mths_since_recent_inq','num_bc_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats',
                 'num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies',
                 'tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','orig_projected_additional_accrued_interest',
                 'term_int','level']
scaler = StandardScaler()
x_pca = scaler.fit_transform(np.array(x_all[no_category_x]))
x_pca = pd.DataFrame(x_pca,columns=[no_category_x])


# ====== principal component analysis (95% explained variance) ======
def pca(df):
    features = list(df.columns)
    X = np.array(df)
    pca = PCA(n_components='mle')
    pca.fit(X)
    variance_ratio = list(pca.explained_variance_ratio_)
    
    cum_ratio = [i for i in range(1,len(variance_ratio)) if sum(variance_ratio[0:i]) <= 0.95]
    if len(cum_ratio) == 0: cum_ratio = [1]
    
    pca_fit = PCA(n_components = (cum_ratio[-1]+1))
    pca_fit.fit(X)
    return features, pca_fit

# fit data and store principal compnenets
features, pca = pca(x_pca)
df = pd.DataFrame()
df['feature'] = list(features)
components = pca.components_
for i in range(len(components)):
    df['pc_'+str(i+1)] = components[i]
print('# pc: ', len(components))
df.to_csv('pc_no_dummy_95%.csv',sep='\t')

# transform features using learned PCA
pca_transformed = pca.transform(np.array(x_pca))
col = ['pc_x_'+str(i+1) for i in range(len(components))]
pca_transformed = pd.DataFrame(pca_transformed,columns=col)
for i in range(len(col)):
    x_all[col[i]] = pca_transformed[col[i]]
x_all.to_csv('data_exclude_y_with_pc1.csv')


# ========== feature selection 1 ==========
# == recursive feature elimination (RFE) ==
rfe_val = [x for x in list(x_all.columns) if (x not in no_category_x)]
x = x_all[rfe_val]
print('# features: ', len(rfe_val))

model = LogisticRegression(n_jobs=-1)
rfe = RFE(model, 1)
rfe = rfe.fit(np.array(x), np.array(y['level_cor']))
col_filter = x.columns[rfe.support_]

feature_importance = pd.DataFrame()
feature_importance['features'] = rfe_val
feature_importance['rfe_rank'] = list(rfe.ranking_)


# ========== feature selection 2 ==========
# ==== random forest feature importance ===
names = x.columns
clf = RandomForestClassifier(n_estimators=100,random_state=123)
clf.fit(np.array(x), np.array(y['level_cor']))

ranking = [0] * len(clf.feature_importances_)
for i, x in enumerate(sorted(range(len(clf.feature_importances_)), key=lambda y: clf.feature_importances_[y])):
    ranking[x] = len(clf.feature_importances_) - i

feature_importance['random_forest_importance'] = list(clf.feature_importances_)
feature_importance['random_forest_rank'] = ranking

# visualize random forest feature importance
# 1. aggregate importance of dummy variables
states = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC',
          'ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
employ_len = ['X..1.year','X0.years','X1.year','X10..years','X2.years','X3.years','X4.years','X5.years','X6.years','X7.years','X8.years','X9.years',]
verification = ['Not.Verified','Source.Verified','Verified']
purpose = ['car','credit_card','debt_consolidation','educational','home_improvement','house','major_purchase','medical','moving','other',
           'Paying.off.my.son.s.dept...','renewable_energy','small_business','vacation','wedding']
application = ['Individual','Joint.App']
home_ownership = ['ANY','MORTGAGE','NONE','OTHER','OWN','RENT']
initial_list_status = ['f','w']
hardship = ['N','Y']
hard_status = ['ACTIVE','BROKEN','COMPLETE','No']

f = [states, employ_len, verification, purpose, application, home_ownership, initial_list_status, hardship, hard_status]
f_name = ['states', 'employ_len', 'verification', 'purpose', 'application', 'home_ownership', 'initial_list_status', 'hardship', 'hard_status']
agg_imp = []
for i in range(len(f)):
    agg_imp.append([f_name[i], None, feature_importance['random_forest_importance'].loc[feature_importance['features'].isin(f[i])].sum(), None])
agg_imp = pd.DataFrame(agg_imp,columns=['features','rfe_rank','random_forest_importance','random_forest_rank'])
feature_importance = feature_importance.append(agg_imp).reset_index(drop=True)

# 2. visualization
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

importances = np.array(feature_importance['random_forest_importance'].loc[(feature_importance['features'].isin(col))|(feature_importance['features'].isin(f_name))])
feat_names = np.array(col + f_name)
indices = np.argsort(importances)[::-1]
fig = plt.figure(figsize=(20,6))
plt.title("Feature importances by Random Forest Classifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=10)
plt.xlim([-1, len(indices)])
plt.show()


# ========== select important features ==========
# ============ store as new data set ============
delete = ['car','credit_card','debt_consolidation','educational','home_improvement','house','major_purchase','medical','moving','other',
          'Paying.off.my.son.s.dept...','renewable_energy','small_business','vacation','wedding','Individual','Joint.App','ANY','MORTGAGE','NONE','OTHER','OWN',
          'RENT','f','w','N','Y','Not.Verified','Source.Verified','Verified','X35000.0', 'X40000.0', 'X44500.0', 'X65000.0','loan_amnt','funded_amnt',
          'funded_amnt_inv','int_rate','installment','annual_inc','dti','delinq_2yrs','fico_range_low','inq_last_6mths','mths_since_last_delinq',
          'mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','out_prncp','out_prncp_inv','total_pymnt', 'total_pymnt_inv',
          'total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt','last_fico_range_high',
          'last_fico_range_low','collections_12_mths_ex_med','tot_coll_amt','tot_cur_bal','max_bal_bc','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m',
          'avg_cur_bal','bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt','mort_acc','mths_since_recent_bc_dlq','mths_since_recent_inq',
          'num_bc_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m',
          'num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit',
          'total_il_high_credit_limit','orig_projected_additional_accrued_interest','term_int','level']

cols = [col for col in x_all.columns if col not in delete]
x_new = x_all[cols]
x_new.to_csv('selected_data.csv',sep='\t')


# ========== visualize feature space ==========
# ====== using processed data and t-SNE =======
# 1. select proper perplexity for t-SNE
ts_data = train_x
ts_data['y'] = train_y
ts_data = ts_data.sample(n=20000).reset_index(drop=True)

y = np.array(ts_data['y'].loc[ts_data['y'].isin([1,2,3,4,5])])
X = np.array(ts_data[col].iloc[ts_data.loc[ts_data['y'].isin([1,2,3,4,5])].index])
orange = y == 1
blue = y == 2
red = y == 3
green = y == 4
purple = y == 5

perplexities = [5, 15, 30, 50, 100]
(fig, subplots) = plt.subplots(1, 5, figsize=(15, 8))
for i, perplexity in enumerate(perplexities):
    ax = subplots[i]
    tsne = TSNE(n_components=2, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[orange, 0], Y[orange,1], c="orange", alpha=0.4)
    ax.scatter(Y[blue, 0], Y[blue, 1], c="b", alpha=0.4)
    ax.scatter(Y[red, 0], Y[red, 1], c="r", alpha=0.4)
    ax.scatter(Y[green, 0], Y[green, 1], c="g", alpha=0.4)
    ax.scatter(Y[purple, 0], Y[purple, 1], c="purple", alpha=0.4)
    ax.legend(['class 1','class 2','class 3','class 4','class 5'])

# 2. visualize feature space using t-SNE with proper parameter
# 3D plot plus 3 side view plots
tsne = TSNE(n_components=3, perplexity=30)
Y = tsne.fit_transform(X)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('3D Visualisation by t-SNE')
ax1.scatter(Y[orange, 0], Y[orange,1], Y[orange,2], c="orange", alpha=0.1)
ax1.scatter(Y[blue, 0], Y[blue, 1], Y[blue, 2], c="b", alpha=0.1)
ax1.scatter(Y[red, 0], Y[red, 1], Y[red, 2], c="r", alpha=0.5)
ax1.scatter(Y[green, 0], Y[green, 1], Y[green, 2], c="g", alpha=0.5)
ax1.scatter(Y[purple, 0], Y[purple, 1], Y[purple, 2], c="purple", alpha=0.3)
ax1.legend(['class 1','class 2','class 3','class 4','class 5'])

ax2 = fig.add_subplot(222)
ax2.set_title('Side view 1')
ax2.scatter(Y[orange, 0], Y[orange,1], c="orange", alpha=0.04)
ax2.scatter(Y[blue, 0], Y[blue, 1], c="b", alpha=0.04)
ax2.scatter(Y[red, 0], Y[red, 1], c="r", alpha=0.2)
ax2.scatter(Y[green, 0], Y[green, 1], c="g", alpha=0.2)
ax2.scatter(Y[purple, 0], Y[purple, 1], c="purple", alpha=0.1)

ax3 = fig.add_subplot(223)
ax3.set_title('Side view 2')
ax3.scatter(Y[orange, 0], Y[orange,2], c="orange", alpha=0.04)
ax3.scatter(Y[blue, 0], Y[blue, 2], c="b", alpha=0.04)
ax3.scatter(Y[red, 0], Y[red, 2], c="r", alpha=0.2)
ax3.scatter(Y[green, 0], Y[green, 2], c="g", alpha=0.2)
ax3.scatter(Y[purple, 0], Y[purple, 2], c="purple", alpha=0.1)

ax4 = fig.add_subplot(224)
ax4.set_title('Side view 3')
ax4.scatter(Y[orange,1], Y[orange,2], c="orange", alpha=0.04)
ax4.scatter(Y[blue, 1], Y[blue, 2], c="b", alpha=0.04)
ax4.scatter(Y[red, 1], Y[red, 2], c="r", alpha=0.2)
ax4.scatter(Y[green, 1], Y[green, 2], c="g", alpha=0.2)
ax4.scatter(Y[purple, 1], Y[purple, 2], c="purple", alpha=0.1)
plt.axis('off')

fig.suptitle('t-SNE Visualisation of Feature Space (perplexity = 30)')
