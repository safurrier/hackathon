import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin


class NumericStringCharacterRemover(TransformerMixin):
    """Take a string column and remove common formatting characters 
    (e.g. 1,000 to 1000 and $10 to 10) and transform to numeric
    
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
        self.no_commas = X[self.columns].replace(to_replace={',','\$', '-'}, value='', regex=True)
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
    
class ColumnNameFormatter(TransformerMixin):
    """ Rename a dataframe's column to underscore, lowercased column names
    
    Parameters
    ----------
    rename_cols : list
        A list of the columns which should be renamed. If not specified, all columns
        are renamed.
    export_mapped_names : boolean
        Flag to return export a csv with the mapping between the old column names
        and new ones
        Default is set to False
    export_mapped_names_path : str
        String export path for csv with the mapping between the old column names
        and new ones. If left null, will export to current working drive with 
        filename ColumnNameFormatter_Name_Map.csv
    """
    
    def __init__(self, rename_cols=None, 
                 export_mapped_names=False,
                 export_mapped_names_path=None):
        self.rename_cols = rename_cols
        self.export_mapped_names = export_mapped_names
        self.export_mapped_names_path = export_mapped_names_path
    
    def _set_renaming_dict(self, X, rename_cols):
        """ Create a dictionary with keys the columns to rename and
            values their renamed versions"""
        
        # If no specific columns to rename are specified, use all
        # in the data
        if not rename_cols:
            rename_cols = X.columns.values.tolist()
            
        import re
        # Create Regex to remove illegal characters
        illegal_chars = r"[\\()~#%&*{}/:<>?|\"-]"
        illegal_chars_regex = re.compile(illegal_chars)
        
        # New columns are single strings joined by underscores where 
        # spaces/illegal characters were
        new_columns = ["_".join(re.sub(illegal_chars_regex, " ", col).split(" ")).lower() 
                                for col 
                                in rename_cols]
        renaming_dict = dict(zip(rename_cols, new_columns))
        return renaming_dict
        
    def fit(self, X, y=None):
        """ Check the logic of the renaming dict property"""
        self.renaming_dict = self._set_renaming_dict(X, self.rename_cols)
        # Assert that the renaming dict exists and is a dictionary
        assert(isinstance(self.renaming_dict, dict))
        # Assert that all columns are in the renaming_dict
        #assert all([column in self.renaming_dict.keys() for column in rename_cols])
        return self
        
    def transform(self, X, y=None):
        """ Rename the columns of the dataframe"""
        if self.export_mapped_names:
            # Create mapping of old column names to new ones
            column_name_df = (pd.DataFrame.from_dict(self.renaming_dict, orient='index')
             .reset_index()
             .rename(columns={'index':'original_column_name', 0:'new_column_name'}))
            
            # If no path specified, export to working directory name with this filename
            if not self.export_mapped_names_path:
                self.export_path = 'ColumnNameFormatter_Name_Map.csv'
            column_name_df.to_csv(self.export_mapped_names_path, index=False)
        
        # Return X with renamed columns
        return X.rename(columns=self.renaming_dict)
    
    def fit_transform(self, X, y=None):
        """Convenience function performing both fit and transform"""
        return self.fit(X).transform(X)  
    
class FeatureLookupTable(TransformerMixin):
    """ Given a feature column and path to lookup table, left join the the data
    and lookup values

    Parameters
    ----------
    feature : list
        A list of the column name(s) of the feature to look up. This is what the lookup table will
        be left joined on.
    lookup_key: list
        A list of the column name(s) the lookup table will be joined on. MUST be in same order
        as 'feature' arg. If none specified will default to the 'feature' argument.
    table_lookup_keep_cols: list
        A list of the columns which should be kept from the joined tables. Default is
        to keep all columns from table
    table_path : str
        The path to the table to use as a lookup
    table_format : str
        The type of file the lookup table is. Currently only accepts
        'csv' (default) or 'pickle'
    merge_as_string: Boolean
        Force the key columns to be cast to string before joining
    add_prefix: str
        A prefix to add onto the new columns added from the lookup table
    add_suffix: str
        A suffix to add onto the new columns added from the lookup table

    """
    import pandas as pd

    def __init__(self, feature=None, lookup_key=None,
                 table_lookup_keep_cols=None, table_path=None,
                 table_format='csv', merge_as_string=False,
                add_prefix=None, add_suffix=None):
        self.feature = feature
        self.table_lookup_keep_cols = table_lookup_keep_cols
        self.table_path = table_path
        self.table_format = table_format
        self.merge_as_string = merge_as_string
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix

        # Determine which column to left join the lookup table on
        if not lookup_key:
            # If none specified use 'feature'
            self.lookup_key = self.feature
        else:
            self.lookup_key = lookup_key

        if self.table_format == 'csv':
            self.lookup_table = pd.read_csv(self.table_path)
        elif self.table_format == 'pickle':
            self.lookup_table =  pd.read_pickle(self.table_path)


    def fit(self, X, y=None):
        # If transformer has already been fit
        # and has fitted_data attribute
        if hasattr(self, 'fitted_data'):
            # Reload the lookup table because column
            # names may have been changd
            if self.table_format == 'csv':
                self.lookup_table = pd.read_csv(self.table_path)
            elif self.table_format == 'pickle':
                self.lookup_table =  pd.read_pickle(self.table_path)

        # Cast dtypes to string if specified
        if self.merge_as_string:
            X[self.feature] = X[self.feature].astype(str)
            self.lookup_table[self.lookup_key] = \
            self.lookup_table[self.lookup_key].astype(str)

        # Determine which columns to keep from lookup table
        # If none specified use all the columns in the lookup table
        if not self.table_lookup_keep_cols:
            all_cols = self.lookup_table.columns.values.tolist()
            all_cols = [col for col in all_cols if col not in self.feature]
            self.table_lookup_keep_cols = all_cols

        # Reduce to only desired columns before merge
        # Rename lookup_key to the same as self.feature
        keep_cols = self.table_lookup_keep_cols + self.lookup_key
        self.lookup_table = self.lookup_table[keep_cols]

        # Renaming dict
        # Creating a renaming dict to rename the lookup
        # table keys to the 'feature' keys
        renaming_dict = dict(zip(self.lookup_key, self.feature))
        self.lookup_table = self.lookup_table.rename(columns=renaming_dict)


        if self.add_prefix:
            # Don't add oprefix to ORIGINAL table lookup key columns
            keep_cols = [col for col in keep_cols if col not in self.lookup_key]
            # Concat the renamed columns (with prefix added) and the key column
            self.lookup_table = pd.concat([self.lookup_table[keep_cols].add_prefix(self.add_prefix),
                                           self.lookup_table[self.feature]], axis=1)
            # Update keep_cols in case adding a suffix also
            keep_cols = self.lookup_table.columns.values.tolist()
            # Remove the key column which is now the SAME as SELF.FEATURE
            keep_cols = [col for col in keep_cols if col not in self.feature]


        if self.add_suffix:
            # Don't add suffix to key columns
            # Remove the key column which is now the SAME as SELF.FEATURE
            keep_cols = [col for col in keep_cols if col not in self.feature]
            # If a prefix has already been added, remove the updated key column
            # that will have the prefix on it.
            
            # Concat the renamed columns (with suffix added) and the key column
            self.lookup_table = pd.concat([self.lookup_table[keep_cols].add_suffix(self.add_suffix),
                                           self.lookup_table[self.feature]], axis=1)

        # Left join the two
        new_df = pd.merge(X,
                self.lookup_table,
                on=self.feature,
                          how='left')

        # Assign to attribute
        self.fitted_data = new_df

        return self

    def transform(self, X, y=None):
        if hasattr(self, 'fitted_data'):
            return self.fitted_data
        else:
            print('Transformer has not been fit yet')

    def fit_transform(self, X, y=None):
        if hasattr(self, 'fitted_data'):
            # Reload the lookup table because column
            # names may have been changd
            if self.table_format == 'csv':
                self.lookup_table = pd.read_csv(self.table_path)
            elif self.table_format == 'pickle':
                self.lookup_table =  pd.read_pickle(self.table_path)

        return self.fit(X, y=y).fitted_data
    
    
    
# Functions for use with FunctionTransformer
def replace_column_values(df, col=None, values=None, replacement=None, new_col_name=None):
    """ Discretize a continuous feature by seperating it into specified quantiles
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    col: str
        The name of the column to replace certain values in 
    values: list
        A list of the values to replace
    replacement: object
        Replaces the matches of values
    new_col_name: str
        The name of the new column which will have the original with replaced values
        If None, the original column will be replaced inplace. 
  
    Returns
    ----------  
    df_copy: Pandas DataFrame
        The original dataframe with the column's value replaced
    """
    # Copy so original is not modified
    df_copy = df.copy()
    if not values:
        return print('Please specify values to replace')
        
    if not replacement:
        return print('Please specify replacement value')
        
    
    # If  column name specified, create new column
    if new_col_name:
        df_copy[new_col_name] = df_copy[col].replace(values, replacement)
    # Else replace old column
    else:
        df_copy[col] = df_copy[col].replace(values, replacement)
    return df_copy

def replace_df_values(df, values):
    """ Call pd.DataFrame.replace() on a dataframe and return resulting dataframe.
    Values should be in format of nested dictionaries, 
    E.g., {‘a’: {‘b’: nan}}, are read as follows: 
        Look in column ‘a’ for the value ‘b’ and replace it with nan
    """
    df_copy = df.copy()
    return df_copy.replace(values)
