import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin


class NumericCommaRemover(TransformerMixin):
    """Take a string column and remove commas and transform to numeric
    
    Parameters
    ----------
    columns : list
        A list of the columns for which to apply this transformation to
    return_numeric : boolean
        Flag to return the comma-removed data as a numeric column. 
        Default is set to true
    """
    def __init__(self, columns, return_numeric=True):
        self.columns = columns
        self.return_numeric = return_numeric

    def fit(self, X, y=None):
        # Within the selected columns, regex search for commas and replace
        # with whitespace 
        self.no_commas = X[self.columns].replace(to_replace=',', regex=True, value='')
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Replace original columns with new columns
        X[self.columns] = self.no_commas
        # If numeric, apply pd.to_numeric
        if self.return_numeric:
            X[self.columns] = X[self.columns].apply(pd.to_numeric)
        return X
    
    def fit_transform(self, X, y=None):
        """Convenience function performing both fit and transform"""
        return self.fit(X).transform(X)