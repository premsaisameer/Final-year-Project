import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std

matches = pd.read_csv("/home/prem/PycharmProjects/SSSIHL-IPL/SAMEER/win_prediction/final_matches_data.csv")
test = pd.read_csv("/home/prem/PycharmProjects/SSSIHL-IPL/datasets/x_test.csv")
print("*******************************************************************************", test)

# To check the number of columns containing null values
null_columns = matches.isnull().sum()
print(null_columns[null_columns > 0])

# Checking any team name appearing twice in the matches data set
print("Unique Team names in the winner column \n", matches['winner'].unique())

# Checking any city is appearing twice in the city column
print("Unique City names in the city column \n", matches['city'].unique())

matches.replace(
    ['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Deccan Chargers', 'Chennai Super Kings',
     'Rajasthan Royals', 'Delhi Daredevils', 'Gujarat Lions', 'Kings XI Punjab',
     'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Rising Pune Supergiant', 'Kochi Tuskers Kerala',
     'Pune Warriors', 'Delhi Capitals']

    , ['MI', 'KKR', 'RCB', 'DC', 'CSK', 'RR', 'DD', 'GL', 'KXIP', 'SRH', 'RPS', 'RPS', 'KTK', 'PW', 'DD'], inplace=True)

matches['winner'].fillna('Draw', inplace=True)

matches.venue.replace({'Feroz Shah Kotla Ground': 'Feroz Shah Kotla',
                       'M Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
                       'MA Chidambaram Stadium, Chepauk': 'M.A. Chidambaram Stadium',
                       'M. A. Chidambaram Stadium': 'M.A. Chidambaram Stadium',
                       'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association Stadium',
                       'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association Stadium',
                       'IS Bindra Stadium': 'Punjab Cricket Association Stadium',
                       'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
                       'Rajiv Gandhi Intl. Cricket Stadium': 'Rajiv Gandhi International Stadium',
                       'ACA-VDCA Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium'}, regex=True,
                      inplace=True)

# imputing the values in column city based on venue
conditions = [matches["venue"] == "Rajiv Gandhi International Stadium",
              matches["venue"] == "Maharashtra Cricket Association Stadium",
              matches["venue"] == "Saurashtra Cricket Association Stadium",
              matches["venue"] == "Holkar Cricket Stadium",
              matches["venue"] == "M. Chinnaswamy Stadium",
              matches["venue"] == "Wankhede Stadium",
              matches["venue"] == "Eden Gardens",
              matches["venue"] == "Feroz Shah Kotla",
              matches["venue"] == "Punjab Cricket Association Stadium",
              matches["venue"] == "Green Park",
              matches["venue"] == "Dr DY Patil Sports Academy",
              matches["venue"] == "Sawai Mansingh Stadium", matches["venue"] == "M.A. Chidambaram Stadium",
              matches["venue"] == "Newlands", matches["venue"] == "St George's Park",
              matches["venue"] == "Kingsmead", matches["venue"] == "SuperSport Park",
              matches["venue"] == "Buffalo Park", matches["venue"] == "New Wanderers Stadium",
              matches["venue"] == "De Beers Diamond Oval", matches["venue"] == "OUTsurance Oval",
              matches["venue"] == "Brabourne Stadium", matches["venue"] == "Sardar Patel Stadium",
              matches["venue"] == "Barabati Stadium",
              matches["venue"] == "Vidarbha Cricket Association Stadium, Jamtha",
              matches["venue"] == "Himachal Pradesh Cricket Association Stadium", matches["venue"] == "Nehru Stadium",
              matches["venue"] == "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
              matches["venue"] == "Subrata Roy Sahara Stadium",
              matches["venue"] == "Shaheed Veer Narayan Singh International Stadium",
              matches["venue"] == "JSCA International Stadium Complex",
              matches["venue"] == "Sheikh Zayed Stadium", matches["venue"] == "Sharjah Cricket Stadium",
              matches["venue"] == "Dubai International Cricket Stadium",
              matches["venue"] == "Feroz Shah Kotla Ground", ]

values = ['Hyderabad', 'Mumbai', 'Rajkot', "Indore", "Bengaluru", "Mumbai", "Kolkata", "Delhi", "Mohali", "Kanpur",
          "Pune", "Jaipur", "Chennai", "Cape Town", "Port Elizabeth", "Durban",
          "Centurion", 'Eastern Cape', 'Johannesburg', 'Northern Cape', 'Bloemfontein', 'Mumbai', 'Ahmedabad',
          'Cuttack', 'Jamtha', 'Dharamshala', 'Chennai', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi',
          'Abu Dhabi', 'Sharjah', 'Dubai', 'Delhi']
matches['city'] = np.where(matches['city'].isnull(),
                           np.select(conditions, values),
                           matches['city'])
null_columns = matches.isnull().sum()
print(null_columns[null_columns > 0])

print(matches[matches['winner'].isna() == True])

# Removing records having null values in "winner" column
matches = matches[matches["winner"].notna()]

null_columns = matches.isnull().sum()
print(null_columns[null_columns > 0])

# encoder
encoder = LabelEncoder()
matches["team1"] = encoder.fit_transform(matches["team1"])
matches["team2"] = encoder.fit_transform(matches["team2"])
matches["winner"] = encoder.fit_transform(matches["winner"].astype(str))
matches["toss_winner"] = encoder.fit_transform(matches["toss_winner"])
matches["venue"] = encoder.fit_transform(matches["venue"])

# outcome variable as a probability of team1 winning
matches.loc[matches["winner"] == matches["team1"], "team1_win"] = 1
matches.loc[matches["winner"] != matches["team1"], "team1_win"] = 0

matches.loc[matches["toss_winner"] == matches["team1"], "team1_toss_win"] = 1
matches.loc[matches["toss_winner"] != matches["team1"], "team1_toss_win"] = 0

matches["team1_bat"] = 0
matches.loc[(matches["team1_toss_win"] == 1) & (matches["toss_decision"] == "bat"), "team1_bat"] = 1

# Writting new Data Frame
# matches.to_csv("final_new_matches_v1.csv")

prediction_df = matches[
    ["team1", "team2", "team1_toss_win", "team1_bat", "team1_win", "venue", "team1_score", "team2_score"]]

# dropping higly correlated features
correlated_features = set()
correlation_matrix = prediction_df.drop('team1_win', axis=1).corr()
pd.set_option("display.max_rows", None, "display.max_columns", None)
print("Correation matrix of the X variable columns \n", correlation_matrix.corr())

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
# mask = np.triu(np.ones_like(correlation_matrix.corr(), dtype=np.bool))
# heatmap = sns.heatmap(correlation_matrix.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':25}, pad=16);
# plt.show()

prediction_df.drop(columns=correlated_features)

# feature selection
X = prediction_df.drop('team1_win', axis=1)
target = prediction_df['team1_win']
target = target.astype(int)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X['team1_score'] = scaler.fit_transform(X[['team1_score']])
X['team2_score'] = scaler.fit_transform(X[['team2_score']])

pd.set_option("display.max_rows", None, "display.max_columns", None)
print("Matches dataset first five rows : \n", X.head())

# Splitting the data into training and testing data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, target, stratify=target, test_size=0.2, random_state=0)

print("\n\n************************************************Model Predictions******************************\n\n")


def cm_analysis(Y_Test, y_pred, labels, algorithm, ymap=None, figsize=(5, 5)):
    """
        Generate matrix plot of confusion matrix with pretty annotations.
        The plot image is saved to disk.
        args:
          Y_Test:    true label of the data, with shape (nsamples,)
          y_pred:    prediction of the data, with shape (nsamples,)
          filename:  filename of figure file to save
          labels:    string array, name the order of class labels in the confusion matrix.
                     use `clf.classes_` if using scikit-learn models.
                     with shape (nclass,).
          ymap:      dict: any -> string, length == nclass.
                     if not None, map the labels & ys to more understandable strings.
                     Caution: original Y_Test, y_pred and labels must align.
          figsize:   the size of the figure plotted.
        """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        Y_Test = [ymap[yi] for yi in Y_Test]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(Y_Test, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.title("CONFUSION MATRIX of " + algorithm)
    plt.show()


# from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_auc_score


def stacking(name, model, train, y, test, ytest, n_fold):
    folds = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)
    test_pred = np.empty((test.shape[0], 1), float)
    train_pred = np.empty((0, 1), float)
    for train_indices, val_indices in folds.split(train, y.values):
        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        model.fit(X=x_train, y=y_train)
        train_pred = np.append(train_pred, model.predict(x_val))
        test_pred = np.append(test_pred, model.predict(test))
    y_pred = model.predict(test)
    # predict probabilities
    pred_prob1 = model.predict_proba(test)
    print("Confusion Matrix \n", confusion_matrix(ytest, y_pred))
    print("Classification Report \n", classification_report(ytest, y_pred))
    print('Accuracy of ' + name + ' classifier on test set: {:.4f}'.format(
        metrics.accuracy_score(ytest, y_pred)))
    print("AUC value :", roc_auc_score(ytest, pred_prob1[:, 1]))
    labels = [0, 1]
    cm_analysis(y_test, y_pred, labels, name)

    return test_pred.reshape(-1, 1), train_pred


'''
# Logistic regression
model1 = LogisticRegression(random_state=7)
test_pred1, train_pred1 = Stacking(name="Decision Tree", model=model1, n_fold=10, train=X_train, test=X_test, y=y_train,
                                   ytest=y_test)

# Decision Tree
model2 = DecisionTreeClassifier(random_state=7)
test_pred1, train_pred1 = Stacking(name="Decision Tree", model=model2, n_fold=10, train=X_train, test=X_test, y=y_train,
                                   ytest=y_test)

'''
models = [('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')),
          ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=7)),
          ('SVM', SVC(gamma='auto', random_state=7, probability=True)), ('KNN', KNeighborsClassifier()),
          ('Decision Tree Classifier', DecisionTreeClassifier(random_state=7)), ('Gaussian NB', GaussianNB())]
for name, model1 in models:
    test_pred1, train_pred1 = stacking(name=name, model=model1, n_fold=10, train=X_train, test=X_test,
                                       y=y_train,
                                       ytest=y_test)

'''train_pred1 = pd.DataFrame(train_pred1)
test_pred1 = pd.DataFrame(test_pred1)'''



# Logistic Regression
logreg = LogisticRegression(random_state=7)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Confusion Matrix \n", confusion_matrix(y_test, y_pred))
print("Classification Report \n", classification_report(y_test, y_pred))
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(X_test, y_test)))
#cm_analysis(y_test, y_pred, labels, "LOGISTIC REGRESSION")


from sklearn.metrics import roc_curve

pred_prob2 = logreg.predict_proba(X_test)

# roc curve for models
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:, 1], pos_label=1)
# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


print("\n\n*********************************************Logistic Regression*******************************************")

'''
# SVM
svm = SVC(random_state=7)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy of SVM classifier on test set: {:.4f}'.format(svm.score(X_test, y_test)))
#cm_analysis(y_test, y_pred, labels, "SVM")

print("\n\n*********************************************SVM*******************************************")

# Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=7)
dtree.fit(X_train, y_train)
dtree.score(X_test, y_test)
y_pred = dtree.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy of decision tree classifier on test set: {:.4f}'.format(dtree.score(X_test, y_test)))
#cm_analysis(y_test, y_pred, labels, "DECISION TREE")

print("\n\n*********************************************DECISION TREE*******************************************")

# Random Forest Classifier
randomForest = RandomForestClassifier(n_estimators=100, random_state=7)
randomForest.fit(X_train, y_train)
randomForest.score(X_test, y_test)
y_pred = randomForest.predict(X_test)
print("Confusion matrix\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy of random forest classifier on test set: {:.4f}'.format(randomForest.score(X_test, y_test)))

'''

# RFC

rf_classifier = RandomForestClassifier(class_weight="balanced", random_state=7)

param_grid = {'n_estimators': [50, 75],
              'min_samples_split': [2, 6, 10],
              'min_samples_leaf': [2, 3, 4],
              'max_depth': [5, 10, 20]}

kfold = KFold(n_splits=10)

grid_obj = GridSearchCV(rf_classifier,
                        return_train_score=True,
                        param_grid=param_grid,
                        cv=kfold)

grid_fit = grid_obj.fit(X_train, y_train)

rf_opt = grid_fit.best_estimator_

print(rf_opt)

y_pred = rf_opt.predict(X_test)
pred_prob1 = rf_opt.predict_proba(X_test)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:, 1], pos_label=1)
# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Random Forest')
# plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC', dpi=300)
plt.show()

# fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

labels = [0, 1]
print(classification_report(y_test, y_pred))
print('Accuracy of random forest classifier on test set: {:.4f}'.format(rf_opt.score(X_test, y_test)))
cm_analysis(y_test, y_pred, labels, "RANDOM FOREST")

print("\n\n*********************************************SVM*******************************************")

print(metrics.accuracy_score(y_test, y_pred))
print("AUC value :", roc_auc_score(y_test, pred_prob1[:, 1]))

