import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#import 

data = pd.read_csv("brca_metabric_clinical_data.tsv", sep="\t")

#print(data)

#print("########################################")


#print(data.columns)

#plt.hist(data["Cancer Type Detailed"])
#plt.figure(figsize=(100,10))
#plt.savefig("breast_cancer_detailed_hist")



#Create dataset based on whether Paitents vital status ==> use as dependent variable
#print(data["Patient's Vital Status"].unique())

data = data[data["Patient's Vital Status"] != 'Died of Other Causes']
data = data.dropna(subset=["Patient's Vital Status"])

#print(data)


#drop columns that wont be used as feature variables
#data = data["HER2 status measured by SNP6"].dropna()
data = data.drop(columns=['Study ID', 'Patient ID', 'Sample ID', 'Cancer Type',
						 'Cancer Type Detailed', 'Cellularity', 'Cohort', 'ER status measured by IHC',
						 'Tumor Other Histologic Subtype', 'Integrative Cluster', 'Number of Samples Per Patient',
						 'Sample Type', 'Sex', 'Overall Survival Status', 'Overall Survival (Months)'])
#print(data)
data = data.dropna(axis=0)

print(data)
#print(data.columns)

#transform categorical features
from sklearn.preprocessing import LabelEncoder
#Categorical columns
cat_cols = ["Type of Breast Surgery", "Chemotherapy", "Pam50 + Claudin-low subtype", "ER Status",
			"HER2 status measured by SNP6", 'HER2 Status', "Hormone Therapy", 'Inferred Menopausal State',
			'Primary Tumor Laterality', 'Oncotree Code', 'PR Status', 
			'Radio Therapy', '3-Gene classifier subtype', "Patient's Vital Status", "Relapse Free Status"]


#Taken from: https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
#Ony the 21/10/2021
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
#print(data)
print(data.columns)

#metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

#GRAPHICAL RESULT FUNCTIONS
def graphical_results(model, X_test, y_test, label, y_pred):
	labels = ["Death", "Living"]
	cm = confusion_matrix(y_test, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	disp.plot()
	disp.ax_.set_title("Confusion matrix "+label)
	plt.tight_layout()
	plt.savefig(label, pad_inches=10)
	plt.clf()

	ax = plt.gca()
	plot_roc_curve(model, X_test, y_test, ax=ax)
	_ = ax.set_title("ROC curve "+label+" ")
	plt.savefig("ROC curve"+label)
	plt.clf()


def results(classifier, y_pred, y_test):
	print(f"{classifier} Results:\n", "roc_auc_score: ", roc_auc_score(y_test, y_pred),"\n",
			"accuracy_score: ", accuracy_score(y_test, y_pred),"\n",
			"f1_score: ", f1_score(y_test, y_pred),"\n",
			"precision_score: ", precision_score(y_test, y_pred),"\n",
			"recall_score: ", recall_score(y_test, y_pred),"\n",
			"matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred),"\n",)



#data = data[cat_cols].apply(LabelEncoder.fit_transform(cat_cols))
data = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(data)
print(data)
#print(data.drop)



#Split the Data up
from sklearn.model_selection import train_test_split
y = data["Patient's Vital Status"].copy()
X = data.drop(columns=["Patient's Vital Status"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#
def feature_importance(model, X_train, y_train, features, model_name):
	model.fit(X_train, y_train)
	imp_1 = model.feature_importances_
	#print(imp_1)
	model.fit(X_train, y_train)
	imp_2 = model.feature_importances_
	#print(imp_2)
	model.fit(X_train, y_train)
	imp_3 = model.feature_importances_
	#print(imp_3)
	imp_average = np.mean( np.array([ imp_1, imp_2, imp_3 ]), axis=0 )
	#print(imp_average)
	plt.bar(range(0,len(imp_average)), imp_average)
	plt.xlabel('Features')
	plt.ylabel('Relative importance')
	plt.title('Relative Importance of Predicitive Features')
	plt.savefig(model_name + "feature importance")
	

#random forect classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
feature_importance(rfc, X_train, y_train, list(X.columns), "Random Forest")
results("Random forest", y_pred, y_test)
graphical_results(rfc, X_test, y_test, "Random forest", y_pred)
#'''

#adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
AdaBoost_clf = AdaBoostClassifier()
AdaBoost_clf.fit(X_train, y_train)
y_pred = AdaBoost_clf.predict(X_test)
results("Adaboost", y_pred, y_test)
graphical_results(AdaBoost_clf, X_test, y_test, "Adaboost", y_pred)



#SVM classifier
from sklearn.svm import SVC
svc = SVC()
svc  = RandomForestClassifier()
svc .fit(X_train, y_train)
y_pred = svc.predict(X_test)
results("Support Vector Machine", y_pred, y_test)
graphical_results(svc, X_test, y_test, "SVM", y_pred)

#recursive feature elimination and cross-validation
from sklearn.feature_selection import RFECV

clf = RFECV(rfc, step=1, cv=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Optimal number of features : %d" % clf.n_features_in_)
print("Feature Ranking:\n", clf.ranking_)
print("The mask of selected features\n", clf.support_)
results("Random forest recursive feature elimination", y_pred, y_test)


#'''
