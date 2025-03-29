from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# List of common "missing" values 
missing_values = {'none', 'missing', 'empty', '?', 'na', 'null', 'undefined', 'unknown', 'n/a'}  

def replace_missing_values(df):
    """
    Replaces common text representations of missing values with np.nan.
    
    Args:
    df (DataFrame): Input dataframe
    
    Returns:
    DataFrame: Modified dataframe with missing values replaced
    """
    return df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip().lower() in missing_values else x)



def fill_missing_values(df_train, df_test, k_neighbors=5):  
    """
    Fills missing values in the numeric columns using KNN imputation, and directly updates the input DataFrames.
    
    Args:
    df_train (DataFrame): Training data with potential missing values
    df_test (DataFrame): Test data with potential missing values
    k_neighbors (int): Number of neighbors to use for imputation (default is 5)
    
    Returns:
    None: The input DataFrames are modified in place
    """
    
    # Select numeric columns from both train and test data
    train_numeric = df_train.select_dtypes(include=['number'])
    test_numeric = df_test.select_dtypes(include=['number'])

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=k_neighbors)

    # Apply KNN imputation to both train and test data
    df_train[train_numeric.columns] = imputer.fit_transform(train_numeric)
    df_test[test_numeric.columns] = imputer.transform(test_numeric)

    return df_train, df_test




def explore_categorical_data(df_train, df_test):  
    """
    A function to explore and check categorical variables for missing values, unique values, 
    and value distributions in both the training and testing datasets.
    
    Args:
    df_train (DataFrame): The training dataset.
    df_test (DataFrame): The testing dataset.
    
    Returns:
    None: Prints results for each step of exploration.
    """
    
    # 1. Select categorical columns (object dtype)
    train_categorical = df_train.select_dtypes(include=['object'])
    test_categorical = df_test.select_dtypes(include=['object'])
    
    # 2. Check for missing values (null values)
    print("Missing values in training set:")
    print(train_categorical.isnull().sum())
    print("\nMissing values in testing set:")
    print(test_categorical.isnull().sum())
    
    # Calculate missing percentage
    missing_percentage_train = train_categorical.isnull().mean() * 100
    missing_percentage_test = test_categorical.isnull().mean() * 100
    
    print("\nMissing percentage in training set:")
    print(missing_percentage_train)
    print("\nMissing percentage in testing set:")
    print(missing_percentage_test)
    
    # 3. Check for the number of unique values in each categorical column
    print("\nNumber of unique values in the training set:")
    print(train_categorical.nunique())
    print("\nNumber of unique values in the testing set:")
    print(test_categorical.nunique())
    
    # 4. Check value counts for each categorical column
    print("\nValue counts for each categorical column in the training set:")
    for col in train_categorical.columns:
        print(f"\n{col} value counts:")
        print(train_categorical[col].value_counts())
    
    print("\nValue counts for each categorical column in the testing set:")
    for col in test_categorical.columns:
        print(f"\n{col} value counts:")
        print(test_categorical[col].value_counts())





def feature_engineering(df):

    """
    Engineers new features. For CodingChallenge, to be applied to TRAIN_subset df, TEST_subset df, and then TEST_df, seperately:  
    - total_sessions
    - user_purchase_rate
    - total_page_views
    - avg_session_duration
    - total_session_duration
    - user_category_encoded
    - browser_purchase_rate
    - device_purchase_rate

    Args:
    df to be engineered. 
    
    Returns:
    None: The input DataFrame is augmented with new features.
    """

    #3(a) numeric values of interest

    #3(a)(I) session_id  
    #Engineer: **total_sessions**  
    df['total_sessions'] = df.groupby('user_id')['session_id'].transform('count')

    # 3(a)(II) abandoned_cart
    #Engineer: **user_purchase_rate**  
    df['total_purchases'] = df.groupby('user_id')['abandoned_cart'].transform(lambda x: (x == 0).sum())
    df['purchase_rate'] = (df['total_purchases'] / df['total_sessions']) * 100

    # 3(a)(III) page_views
    #Engineer: **total_page_views**

    df['total_page_views'] = df.groupby('user_id')['page_views'].transform('sum')

    # 3(a)(IV) session_duration
    #Engineer: **avg_session_duration**

    df['avg_session_duration'] = df.groupby('user_id')['session_duration'].transform('mean')

    # 3(a)(V) session_duration
    #Engineer: **total_session_duration**

    df['total_session_duration'] = df.groupby('user_id')['session_duration'].transform('sum')



    # 3(b) categoric values of interest

    # 3(b)(I) user_category
    #Encode engineer: **user_category_encoded** (ordinal)  

    user_category_values = {'new_user': 1, 'recurring_user': 2, 'premium_user': 3}
    df['user_category_encoded'] = df['user_category'].map(user_category_values).fillna(0).astype(int)

    # 3(b)(II) browser 
    #Encode engineer: **browser_purchase_rate**
    purchase_rate_by_browser = df.groupby('browser')['abandoned_cart'].apply(lambda x: (x == 0).sum() / len(x) * 100)
    df['browser_purchase_rate'] = df['browser'].map(purchase_rate_by_browser)

    # 3(b)(III) device_type 
    #Encode engineer: **device_purchase_rate**
    purchase_rate_by_device = df.groupby('device_type')['abandoned_cart'].apply(lambda x: (x == 0).sum() / len(x) * 100)
    df['device_purchase_rate'] = df['device_type'].map(purchase_rate_by_device)


    return df




def marketing_target(df):
    """
    Assigns a marketing_target category based on total engagement score using percentiles.
    
    Args:
    df (DataFrame): The dataset containing engineered engagement features.

    Returns:
    DataFrame: The modified DataFrame with a new 'marketing_target' column.
    """

    # Define the feature columns to sum
    columns = ['total_sessions', 'user_purchase_rate', 'total_page_views', 
               'avg_session_duration', 'total_session_duration', 
               'user_category_encoded', 'browser_purchase_rate', 'device_purchase_rate']
    
    # Calculate the engagement total per row
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    df['marketing_total'] = df[columns].sum(axis=1)
    df['marketing_total'] = df[columns].sum(axis=1)

    # Calculate IQR thresholds
    Q1 = df['marketing_total'].quantile(0.33)
    Q3 = df['marketing_total'].quantile(0.66)

    # Assign categories based on quartiles
    df['marketing_target'] = np.where(df['marketing_total'] < Q1, 1,  
                                      np.where(df['marketing_total'] < Q3, 2, 3))

    # Drop marketing_total column after assigning categories
    df.drop(columns=['marketing_total'], inplace=True)
    
    return df