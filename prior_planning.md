# Imports
    import pandas as pd   
    import numpy as np  

    from sklearn.preprocessing import MinMaxScaler  
    from sklearn.model_selection import train_test_split  
    from sklearn.impute import KNNImputer



# 1. Convert TRAIN Folder Data user_data.csv and session_data.csv Dataframes 
    TRAIN_user_data_route = ''  
    TRAIN_session_data_route = ''

    TRAIN_user_data_df = pd.read_csv(TRAIN_user_data_route)  
    TRAIN_session_data_df = pd.read_csv(TRAIN_session_data_route)

- TRAIN_user_data_df 
- TRAIN_session_data_df




# 2. Join TRAIN_user_data_df to TRAIN_session_data_df on user_id 

    df = pd.merge(TRAIN_session_data_df, TRAIN_user_data_df, on="user_id", how="left")



## 2(a). Exploratory Analysis

### 2(a)(I). DataTypes
    df.info()

### 2(a)(II). Missing Values
replace faux missing string values with nan  

    replace_missing_values(df)  

**function** in utils.py


### 2(a)(II). Check distributions
    df.select_dtypes(include=['number']).describe()


## 2(b). Split 80:20 TRAIN_subset:TEST_subset avoid data leakage in steps 2(c) and 3. 

Combine the categorical columns into one for stratification  

    df['stratify_col'] = df[['user_category', 'device_type', 'browser_type']].astype(str).agg('-'.join, axis=1)  

Split the data  

    TRAIN_subset, TEST_subset = train_test_split(df, test_size=0.2, random_state=42, stratify=df['stratify_col'])

Drop the temporary stratification column  

    TRAIN_subset.drop('stratify_col', axis=1, inplace=True)
    TEST_subset.drop('stratify_col', axis=1, inplace=True)




## 2(c). Fill missing values: 

### 2(c)(II). Numerical Values: KNN impute 

**find and impute**

    df_train_imputed, df_test_imputed = fill_missing_values(TRAIN_subset, TEST_subset, k_neighbors=5) 

**function** in utils.py      




### 2(c)(II). Categorical Varables 

Will first explore what kind of nulls (if any) there are, and whether they affect the key variables:

**key categorical variables**  

    df[['user_category', 'device_type', 'browser']]

**Checking**  

    explore_categorical_data(TRAIN_subset, TEST_subset)

**function** in utils.py



# 3. Feature Engineering in TRAIN_subset (and TEST_subset).

    feature_engineering(df)  

**Function** in utils.py.   
Apply _seperately_ to TRAIN(TRAIN_subset, TEST_subset), and TEST.



# 4. Create Target column, marketing_target, for TRAIN data: 1 (low), 2 (medium), and 3 (high). Apply to TRAIN_subset and TEST_subset

    marketing_target(df)

**function** in utils.py


## 4(a). Check distribution of marketing_target. 



### 4(a)(I). Does marketing_target distribution influence model selection and parameters? 




# 5. Check correlation between variables and marketing_target.




# 6. Feature Selection
    model_features = ['', '']  
    X_train = TRAIN_subset[model_features]  
    y_train = TRAIN_subset['marketing_target']  
    X_test = TEST_subset[model_features]  
    y_test = TEST_subset['marketing_target']  


## 6(a). Check sizes
    print('X_train shape:', X_train.shape)  
    print('y_train shape:', y_train.shape)  
    print('X_test shape:', X_test.shape)  
    print('y_test shape:', y_test.shape)




# 7. Train Multi-Class Classifier on TRAIN_subset




# 8. Test Model TEST_subset




# 9. Model performance metrics: F1 score




# 10. Create function to apply to TEST Folder Data




## 11. JSON format
test_id and marketing_target, for example,

{
    "target": {  
        "297": 1,  
        "11": 3,  
        "67": 3,  
        "54": 3,  
        "156": 2,  
        "290": 2,  
        "193": 3,  
        ...  
  }
}











