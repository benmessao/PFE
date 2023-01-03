import pandas as pd 
import os 
import numpy as np
import pylab as plt


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.compose import make_column_selector as selector
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Read the dataset
#merged1 = pd.read_csv(os.path.join('../data/1416', 'merged1.csv'))
#merged2 = pd.read_csv(os.path.join('../data/1416', 'merged2.csv'))
#merged3 = pd.read_csv(os.path.join('../data/1416', 'merged3.csv'))
#merged4 = pd.read_csv(os.path.join('../data/1416', 'merged4.csv'))
merged5 = pd.read_csv(os.path.join('../data/1416', 'merged5.csv'))

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

data = swap_columns(merged5, 'labelRec', 'estAoA')
data.dropna(inplace=True) # suppression des lignes NAN

X = abs(data.iloc[:,0:83]) # On prend la valeur absolue car les valeurs doivent être positives pour utiliser la méthode Chi2
y = data.iloc[:,-1]

#Remplacer les NAN par 0
#X = [0 if pd.isnull(i) else i for i in X]
#X["deltaHed1"] = X["deltaHed1"].fillna(0,inplace=True) 
#X["deltaHed2"] = X["deltaHed2"].fillna(0,inplace=True)
print(str(len(X)), "nombre de lignes?")

def selection_chi2(X, y):
    # Méthode du Chi 2 pour la sélection de feature

    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(X,y)
    chi_support = fit.get_support()
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # Concaténation
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']

    # affichage plot + text
    featureScores.nlargest(20, 'Score').plot(kind='barh')
    plt.show()

    print(featureScores.nlargest(20,'Score'))  # 17 meilleures caractéristiques

    return chi_support, featureScores

def selection_RF(X, y):
    # Feature importance

    model = ExtraTreesClassifier()
    model.fit(X,y)

    print(model.score(X,y), "means accuracy")
    print(str(len(model.feature_importances_)),"selected features")
    #print(model.feature_importances_) #utiliser la classe intégrée feature_importances des classificateurs basés sur les arbres.
    #tracer un graphique des importances des caractéristiques pour une meilleure visualisation.
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    embeded_rf_support = [True if i in feat_importances.nlargest(20).index else False for i in X.columns.tolist()]

    # affichage plot + text
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()

    print(feat_importances.nlargest(20).index) 

    return embeded_rf_support, feat_importances

def selection_pearson_correlation(X, y, num_feats=20):
    cor_list = []
    feature_name = X.columns.tolist()
    
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)

    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]

    # affichage text
    print(str(len(cor_feature)), 'selected features')
    print(cor_feature)

    return cor_support, cor_feature

def selection_recurcive_feature(X, y):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=20, step=10, verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()

    # affichage text
    print(str(len(rfe_feature)), 'selected features')
    print(rfe_feature)

    return rfe_support, rfe_feature

chi_support = selection_chi2(X,y)
embeded_rf_support = selection_RF(X,y)
cor_support = selection_pearson_correlation(X,y)
rfe_support = selection_recurcive_feature(X,y)

def feature_selection():
    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature': X.columns.tolist(), 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support,
                                        'Random Forest':embeded_rf_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    feature_selection_df.head(84)
    feature_selection_df.to_csv(os.path.join('./1416', 'selection.csv'), index=False)
    