import datetime
import numpy as np
import pandas as pd

# ---------- scikit-learn ------------
# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
# ------------------------------------- 

class CategoricalPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mode_imputer = SimpleImputer(strategy="most_frequent")
        self.cat_cols = ['home_ownership', 'purpose', 'addr_state', 'initial_list_status']
        self.target_encoder = TargetEncoder(handle_missing='return_nan', handle_unknown='return_nan')
    
    def fit(self, X, y=None):
        self.mode_imputer.fit(X[self.cat_cols])
        self.target_encoder.fit(X["zip_code"], y)
        return self
    
    def transform(self, X, y=None):
        Xc = X.copy()
        
       # encode emp_length
        lookup = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, 
                  '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years':10}
        Xc["emp_length"] = Xc["emp_length"].replace(lookup)

        # issue date
        Xc["issue_d"] = pd.to_datetime(Xc["issue_d"])
        tmp = Xc["issue_d"].values # keep a copy of the raw date for when we transform earliest credit line
        Xc["issue_d"] = (Xc["issue_d"]-datetime.datetime(2000,1,1)).astype('timedelta64[M]')

        # earliest credit line
        Xc["earliest_cr_line"] = pd.to_datetime(Xc["earliest_cr_line"])
        Xc["earliest_cr_line"] = (tmp - Xc["earliest_cr_line"]).astype('timedelta64[M]')
        
        # imputation for home_ownership, purpose, addr_state, and initial_list_status
        Xc[self.cat_cols] = self.mode_imputer.transform(Xc[self.cat_cols])
        
        # encode zip code
        Xc["zip_code"] = self.target_encoder.transform(Xc["zip_code"])
        
        return Xc
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, max_corr=1, k_num=1, k_cat=1):
        self.max_corr = max_corr
        self.k_num = k_num
        self.k_cat = k_cat
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.num_cols = None
        self.cat_cols = None
        self.cat_cols_enc = None
        self.all_cols = None
        
    def fit(self, X, y=None):
        Xc = X.copy()
        yc = y.copy()
        
        # identify numeric and categorical columns
        self.num_cols = Xc.columns[Xc.dtypes!="object"].values
        self.cat_cols = Xc.columns[Xc.dtypes=="object"].values
        
        # one-hot-encode categorical columns and convert back to full dataframe
        Xc_cat = self.ohe.fit_transform(Xc[self.cat_cols]).toarray()
        self.cat_cols_enc = self.ohe.get_feature_names(input_features=self.cat_cols)
        Xc_cat = pd.DataFrame(Xc_cat, columns = self.cat_cols_enc)
        Xc_num = Xc[self.num_cols]
        Xc = pd.concat([Xc_cat, Xc_num], axis=1)
        
        # handling multicollinearity
        if self.max_corr < 1:
            X_corr = Xc.corr()
            X_not_correlated = ~(X_corr.mask(np.tril(np.ones([len(X_corr)]*2, dtype=bool))).abs() > self.max_corr).any() 
            un_corr_idx = X_not_correlated.loc[X_not_correlated[X_not_correlated.index]].index.values
            Xc = Xc[un_corr_idx]
            self.cat_cols_enc = np.array([c for c in Xc.columns if c in self.cat_cols_enc])
            self.num_cols = np.array([c for c in Xc.columns if c in self.num_cols])
        
        # numeric feature selection
        n_feats = int(self.k_num*len(self.num_cols))
        if self.k_num < 1 and n_feats > 0:
            num_selector = SelectKBest(f_classif, k=n_feats)
            num_selector.fit(Xc[self.num_cols], yc)
            self.num_cols = self.num_cols[num_selector.get_support()]
            Xc = Xc[self.cat_cols_enc.tolist() + self.num_cols.tolist()]
        
        # categorical feature selection
        n_feats = int(self.k_cat*len(self.cat_cols_enc))
        if self.k_cat < 1 and n_feats > 0:
            cat_selector = SelectKBest(lambda X,y: mutual_info_classif(X, y, discrete_features=True), k=n_feats)
            cat_selector.fit(Xc[self.cat_cols_enc], yc)
            self.cat_cols_enc = self.cat_cols_enc[cat_selector.get_support()]
            Xc = Xc[self.cat_cols_enc.tolist() + self.num_cols.tolist()]
        
        # all selected features
        self.all_cols = self.cat_cols_enc.tolist() + self.num_cols.tolist()
        
        # fit numeric scaler
        self.scaler.fit(Xc[self.all_cols])
                
        return self
    
    def transform(self, X, y=None):
        Xc = X.copy()
        Xc_cat = self.ohe.transform(Xc[self.cat_cols]).toarray()
        Xc_cat = pd.DataFrame(Xc_cat, columns = self.ohe.get_feature_names(input_features=self.cat_cols))
        Xc_num = Xc[self.num_cols]
        Xc = pd.concat([Xc_cat, Xc_num], axis=1)
        Xc = pd.DataFrame(self.scaler.transform(Xc[self.all_cols]), columns = self.all_cols)
        
        return Xc
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)
        
class NumericalPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_imputer = SimpleImputer(strategy="median")
    
    def fit(self, X, y=None):
        self.num_cols = X.columns[X.dtypes!="object"]
        self.mo_sin_cols = [x for x in X.columns if "mo_sin" in x or "mths_since" in x]
        self.other_num_cols = [x for x in self.num_cols if x not in self.mo_sin_cols]
        self.median_imputer.fit(X[self.other_num_cols])
        return self
    
    def transform(self, X, y=None):
        Xc = X.copy()
        
        for col in self.mo_sin_cols:
            Xc[col] = Xc[col].apply(lambda x: 0 if np.isnan(x) else 1/max(x,1))
            
        Xc[self.other_num_cols] = self.median_imputer.transform(Xc[self.other_num_cols])
        
        return Xc
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
