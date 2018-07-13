import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin


###############################################################################################################
# Custom Transformers
###############################################################################################################


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
    
    
class TargetAssociatedFeatureValueAggregator(TransformerMixin):
    """ Given a dataframe, a set of columns and associative target thresholds,
        mine feature values associated with the target class that meet said thresholds.
        Aggregate those features values together and create a one hot encode feature when
        a given observation matches any of the mined features' values.
        
        Currently only works for binary classifier problems where the positive class is
        1 and negative is 0
        
        Example:

    """
    
    def __init__(self, include=None, exclude=None, prefix=None, suffix=None,
                 aggregation_method='aggregate',
                 min_mean_target_threshold=0, min_sample_size=0,
                 min_sample_frequency=0, min_weighted_target_threshold=0,
                 ignore_binary=True):
        
        self.aggregation_method = aggregation_method
        self.include = include
        self.exclude = exclude
        self.prefix = prefix
        self.suffix = suffix
        self.min_mean_target_threshold = min_mean_target_threshold
        self.min_sample_size = min_sample_size
        self.min_sample_frequency = min_sample_frequency
        self.min_weighted_target_threshold = min_weighted_target_threshold
        self.ignore_binary = ignore_binary
        """ 
        Parameters
        ----------
        aggregation_method: str --> Default = 'aggregate'
            Option flag.
            'aggregate'
                Aggregate feature values from specific a specific feature. 
                Aggregate those features values together and create a one hot encode 
                feature when a given observation matches any of the mined features' values.
            'one_hot'
                Create one hot encoded variable for every feature value that meets the 
                specified thresholds.
                Warning: Can greatly increase dimensions of data
        include: list
            A list of columns to include when computing
        exclude: list
            A list of columns to exclude when computing   
        prefix: str
            A string prefix to add to the created columns
        suffix: str
            A string suffix to add to the created columns        
        min_mean_target_threshold : float
            The minimum value of the average target class to use as cutoff.
            E.g. .5 would only return values whose associate with the target is 
            above an average of .5
        min_sample_size: int
            The minimum value of the number of samples for a feature value
            E.g. 5 would only feature values with at least 5 observations in the data
        min_weighted_target_threshold : float
            The minimum value of the frequency weighted average target class to use as cutoff.
            E.g. .5 would only return values whose associate with the frequency weighted target 
            average is above an average of .5
        min_sample_frequency: float
            The minimum value of the frequency of samples for a feature value
            E.g. .5 would only include feature values with at least 50% of the values in the column 
        ignore_binary: boolean
            Flag to ignore include feature values in columns with binary values [0 or 1] as this is 
            redundant to aggregate.
            Default is True
        """
    
    def fit(self, X, y):
        """ 
        Parameters
        ----------
        X: Pandas DataFrame
            A dat
        y: str/Pandas Series of Target
            The STRING column name of the target in dataframe X 
        """
        self.X = X
        
        # Check if y is a string (column name) or Pandas Series (target values)
        if isinstance(y, str):
            self.y = y
        if isinstance(y, pd.Series):
            self.y = y.name
        self.one_hot_dict = df_feature_vals_target_association_dict(X, y,
                                include=self.include, exclude=self.exclude, 
                                min_mean_target_threshold=self.min_mean_target_threshold,
                                min_sample_size=self.min_sample_size,
                                min_sample_frequency=self.min_sample_frequency,
                                min_weighted_target_threshold=self.min_weighted_target_threshold,
                                ignore_binary=self.ignore_binary                                                              
                                )
        return self
    
    def transform(self, X, y=None):
        if not hasattr(self, 'one_hot_dict'):
            return f'{self} has not been fitted yet. Please fit before transforming'
        if self.aggregation_method == 'one_hot':
            return get_specific_dummies(col_map = self.one_hot_dict, 
                                        prefix=self.prefix, suffix=self.suffix)
        else:
            assert self.aggregation_method == 'aggregate'
            return get_text_specific_dummies(X, col_map = self.one_hot_dict, 
                                             prefix=self.prefix, suffix=self.suffix)

###############################################################################################################
# Functions used with Transformers
###############################################################################################################

def column_values_target_average(df, feature, target,
                                      sample_frequency=True,
                                      freq_weighted_average=True,
                                      min_mean_target_threshold = 0, 
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0):
    """ Group by a feature and computing the average target value and sample size
    Returns a dictionary Pandas DataFrame fitting that criteria

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe where data resides
    feature : str
        Column name for which to groupby and check for average target value
    target : str
        Column name of the target to find grouped by average of
    sample_frequency: Boolean
        Flag to include sample frequency for a given feature value.
        Default is true
    freq_weighted_average: Boolean
        Flag to include the frequency weighted average for a given feature value.
        Default is true        
    min_mean_target_threshold : float
        The minimum value of the average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the target is 
        above an average of .5
    min_sample_size: int
        The minimum value of the number of samples for a feature value
        E.g. 5 would only feature values with at least 5 observations in the data
    min_weighted_target_threshold : float
        The minimum value of the frequency weighted average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the frequency weighted target 
        average is above an average of .5
    min_sample_frequency: float
        The minimum value of the frequency of samples for a feature value
        E.g. .5 would only include feature values with at least 50% of the values in the column         

    Returns
    -------
    grouped_mean_target_df
        DataFrame of the feature values and their asssociations
    """
    grouped_mean_target_df = (df.groupby(by=feature)
     .agg({target:['size', 'mean']})
     .loc[:, target]
     .reset_index()
     .sort_values(by='mean', ascending=False)
     .rename(columns={'mean':'avg_target', 'size':'sample_size'})
    )
    # Sum the sample sizes to get total number of samples
    total_samples = grouped_mean_target_df['sample_size'].sum()
    
    # Flags for adding sample frequency and frequency weighted average
    if sample_frequency:
        # Compute frequency
        grouped_mean_target_df['feature_value_frequency'] = grouped_mean_target_df['sample_size'] / total_samples
        # Filter out minimums
        grouped_mean_target_df = grouped_mean_target_df[grouped_mean_target_df['feature_value_frequency'] >= min_sample_frequency]
        
    if freq_weighted_average:
        # Sample frequency must be calculated for frequency weighted average
        grouped_mean_target_df['feature_value_frequency']  = grouped_mean_target_df['sample_size'] / total_samples 
        grouped_mean_target_df['freq_weighted_avg_target'] = grouped_mean_target_df['feature_value_frequency']  * grouped_mean_target_df['avg_target']
        grouped_mean_target_df = grouped_mean_target_df[(grouped_mean_target_df['feature_value_frequency'] >= min_sample_frequency)
                                                       & (grouped_mean_target_df['freq_weighted_avg_target'] >= min_weighted_target_threshold)
                                                       ]
        
        # If sample frequency not included, drop the column
        if not sample_frequency:
            grouped_mean_target_df.drop(labels=['feature_value_frequency'], axis=1, inplace=True)
    
    # Filter out minimum metrics
    grouped_mean_target_df = grouped_mean_target_df[
        (grouped_mean_target_df['avg_target'] >= min_mean_target_threshold) 
        & (grouped_mean_target_df['sample_size'] >= min_sample_size)]
    

    
    
    return grouped_mean_target_df



def df_feature_values_target_average(df, target,
                                                           include=None,
                                                           exclude=None,
                                      min_mean_target_threshold = 0, 
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0):
    
    """ For a given dataframe and a target column, groupby each column and compute 
    for each column value the the average target value, feature value sample size,
    feature value frequency, and frequency weighted average target value

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe where data resides
    target : str
        Column name of the target to find grouped by average of
    sample_frequency: Boolean
        Flag to include sample frequency for a given feature value.
        Default is true
    include: list
        A list of columns to include when computing
    exclude: list
        A list of columns to exclude when computing        
    freq_weighted_average: Boolean
        Flag to include the frequency weighted average for a given feature value.
        Default is true        
    min_mean_target_threshold : float
        The minimum value of the average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the target is 
        above an average of .5
    min_sample_size: int
        The minimum value of the number of samples for a feature value
        E.g. 5 would only feature values with at least 5 observations in the data
    min_weighted_target_threshold : float
        The minimum value of the frequency weighted average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the frequency weighted target 
        average is above an average of .5
    min_sample_frequency: float
        The minimum value of the frequency of samples for a feature value
        E.g. .5 would only include feature values with at least 50% of the values in the column         

    Returns
    -------
    feature_values_target_average_df
        DataFrame of the feature values and their asssociations
    """
    
    # Start with all columns and filter out/include desired columns
    columns_to_check = df.columns.values.tolist()
    if include:
        columns_to_check = [col for col in columns_to_check if col in include]
    if exclude:
        columns_to_check = [col for col in columns_to_check if col not in exclude]
        
    # Compute for all specified columns in dataframe
    dataframe_lists = [column_values_target_average(df, column, target,  
                                      min_mean_target_threshold = min_mean_target_threshold, 
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold)
                     .rename(columns={column:'feature_value'}).assign(feature = column)
            for column in columns_to_check if column != target] 
    
    feature_values_target_average_df = pd.concat(dataframe_lists)
    
    return feature_values_target_average_df

def feature_vals_target_association_dict(df, feature, target,  
                                      min_mean_target_threshold = 0, 
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0, 
                                         ignore_binary=True):
    """Return a dictionary of the form column_name:[list of values] for values in a
       feature that have an above certain threshold for feature value mean target value,
       feature value sample size, feature value sample frequency and feature value frequency
       weighted mean target value
       
    """
    if ignore_binary:
        # Check to see if only values are 1 and 0. If so, don't compute rest
        if df[feature].dropna().value_counts().index.isin([0,1]).all():
            return {feature: []}
        
    grouped_mean_target = column_values_target_average(df, feature, target,  
                                      min_mean_target_threshold = min_mean_target_threshold, 
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold)
    
    return {feature: grouped_mean_target[feature].values.tolist()}
    

def df_feature_vals_target_association_dict(df, target,
                                                           include=None,
                                                           exclude=None,
                                      min_mean_target_threshold = 0, 
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0,
                                           ignore_binary=True):

    
    columns_to_check = df.columns.values.tolist()
    if include:
        columns_to_check = [col for col in columns_to_check if col in include]
    if exclude:
        columns_to_check = [col for col in columns_to_check if col not in exclude]
        
    # Compute for all specified columns in dataframe
    list_of_dicts = [feature_vals_target_association_dict(df, column, target,  
                                       min_mean_target_threshold = min_mean_target_threshold, 
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold,
                                                         ignore_binary=ignore_binary)
            for column in columns_to_check if column != target]

    # Combine into single dictionary if there are any values
    # that fit the minimum thresholds
    combined_dict = {}
    for dictionary in list_of_dicts:
        # Check it see if any values in list
        feat_vals = list(dictionary.values())
        if len(feat_vals[0]) >=1:
            combined_dict.update(dictionary)
    return combined_dict


def get_specific_dummies(df, col_map=None, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, one hot the values
    in the column and concat to dataframe. Optional arguments to add prefixes 
    and/or suffixes to created column names.
    
    Example col_map: {'foo':['bar', 'zero']} would create one hot columns 
    for the values bar and zero that appear in the column foo"""
    one_hot_cols = []
    for column, value in col_map.items():
        for val in value:
            # Create one hot encoded arrays for each value specified in key column
            one_hot_column = pd.Series(np.where(df[column] == val, 1, 0))
            # Set descriptive name
            one_hot_column.name = column+'_==_'+str(val)
            # add to list of one hot columns
            one_hot_cols.append(one_hot_column)
    # Concatenate all created arrays together        
    one_hot_cols = pd.concat(one_hot_cols, axis=1)
    if prefix:
        one_hot_cols = one_hot_cols.add_prefix(prefix)
    if suffix:
        one_hot_cols = one_hot_cols.add_suffix(suffix)        
    if return_df:
        return pd.concat([df, one_hot_cols], axis=1)
    else:
        return one_hot_cols
    
def one_hot_column_text_match(df, column, text_phrases, case=False):
    """Given a dataframe, text column to search and a list of text phrases, return a binary
       column with 1s when text is present and 0 otherwise
    """
    # Ignore regex group match warning
    import warnings
    warnings.filterwarnings("ignore", 'This pattern has match groups')

    # Create regex pattern to match any phrase in list

    # The first phrase will be placed in its own groups
    regex_pattern = '({})'.format(text_phrases[0])

    # If there's more than one phrase
    # Each phrase is placed in its own group () with an OR operand in front of it |
    # and added to the original phrase
    
    if len(text_phrases) > 1:
        subsquent_phrases = "".join(['|({})'.format(phrase) for phrase in text_phrases[1:]])
        regex_pattern += subsquent_phrases
        
    # Cast to string to ensure .str methods work
    df_copy = df.copy()
    df_copy[column] = df_copy[column].astype(str)
    
    matches = df_copy[column].str.contains(regex_pattern, na=False, case=case).astype(int)
    
    # One hot where match is True (must use == otherwise NaNs throw error)
    #one_hot = np.where(matches==True, 1, 0 )
    
    return matches
    
def get_text_specific_dummies(df, col_map=None, case=False, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, search for text matches
    for the phrases in the list. Optional arguments to add prefixes 
    and/or suffixes to created column names.
    
    Example col_map: {'foo':['bar', 'zero']} would search the text in the values of
    'foo' for any matches of 'bar' OR 'zero' the result is a one hot encoded
    column of matches"""
    one_hot_cols = []
    for column, value in col_map.items():
        # Create one hot encoded arrays for each value specified in key column
        one_hot_column = pd.Series(one_hot_column_text_match(df, column, value, case=case))
        # Check if column already exists in df
        if column+'_match_for: '+str(value)[1:-1].replace(r"'", "") in df.columns.values.tolist():
            one_hot_column.name = column+'_supplementary_match_for: '+str(value)[1:-1].replace(r"'", "")
        else:
            # Set descriptive name
            one_hot_column.name = column+'_match_for: '+str(value)[1:-1].replace(r"'", "")
        # add to list of one hot columns
        one_hot_cols.append(one_hot_column)
    # Concatenate all created arrays together        
    one_hot_cols = pd.concat(one_hot_cols, axis=1)
    if prefix:
        one_hot_cols = one_hot_cols.add_prefix(prefix)
    if suffix:
        one_hot_cols = one_hot_cols.add_suffix(suffix)       
    if return_df:
        return pd.concat([df, one_hot_cols], axis=1)
    else:
        return one_hot_cols    
    
       
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

    
    
    

    
    
    
    
