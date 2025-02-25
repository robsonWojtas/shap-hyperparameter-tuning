#########################################################################
#     GENERATE PLOTS WITH RANGES ON THEM - run after experiments    #####
#########################################################################
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

setup = {'mode': 'svm', 'kernel': 'linear', 'data': 'adult'}
tuning_filename = 'strojenie_' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random_ranges.png'
best_filename = 'najlepsze' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random_ranges.png'

with open("./masters_utils/workspaces/xgb_adult_workspace.pkl", 'rb') as f:
    data = pickle.load(f)
results = data[0]
rand_search_results = data[3]
iteration = data[6]
best_shap_list = data[15]
best_rand_list = data[16]
plt.plot(list(range(0, iteration)), results, color='b', label='SHAP')
plt.plot(list(range(0, iteration)), rand_search_results, color='r', label='Random')
major_xticks = np.arange(0, iteration, 10)
minor_xticks = np.arange(0, iteration, 1)

major_yticks = np.arange(0, 1.1, 0.1)
minor_yticks = np.arange(0, 1.1, 0.05)

plt.xticks(major_xticks)
plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(minor_xticks))

plt.yticks(major_yticks)
plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(minor_yticks))

plt.grid(True, which='both')

ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
plt.xlabel("Iteracja")
plt.ylabel("Celność", labelpad=5)
plt.legend(loc='lower right')
ax.tick_params(axis='y', pad=15)

# XGB
plt.axvspan(0,7, color='blue', alpha=0.05)
plt.axvspan(7,15, color='blue', alpha=0.15)
plt.axvspan(15,27, color='blue', alpha=0.05)
plt.axvspan(27,41, color='blue', alpha=0.15)
plt.axvspan(41,48, color='blue', alpha=0.05)

ax.text(3, 0.05,'learning_rate', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(11, 0.05,'n_estimators', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(21, 0.05,'max_depth', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(34, 0.05,'min_child_weight', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(45, 0.2,'gamma', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# linear
plt.axvspan(0,10, color='blue', alpha=0.05)
plt.axvspan(10,21, color='blue', alpha=0.15)
plt.axvspan(21,28, color='blue', alpha=0.05)
ax.text(3, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(11, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(21, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# poly
plt.axvspan(0,7, color='blue', alpha=0.05)
plt.axvspan(7,15, color='blue', alpha=0.15)
plt.axvspan(15,25, color='blue', alpha=0.05)
plt.axvspan(25,36, color='blue', alpha=0.15)
plt.axvspan(36,43, color='blue', alpha=0.05)
ax.text(3, 0.05,'degree', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(11, 0.05,'coef0', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(20, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(31, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(39, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# rbf
plt.axvspan(0,8, color='blue', alpha=0.05)
plt.axvspan(8,19, color='blue', alpha=0.15)
plt.axvspan(19,27, color='blue', alpha=0.05)
ax.text(4, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(13, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(23, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# sigmoid
plt.axvspan(0,9, color='blue', alpha=0.05)
plt.axvspan(9,20, color='blue', alpha=0.15)
plt.axvspan(20,31, color='blue', alpha=0.05)
plt.axvspan(31,39, color='blue', alpha=0.15)
ax.text(4, 0.05,'coef0', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(14, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(25, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
ax.text(35, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)


plt.savefig(tuning_filename, dpi=300)

plt.show()