import pandas as pd
import numpy as np

def impute(df, id_col, imputer):
    print('Imputing missing data.')
    
    df_imp = imputer.transform(df)
    index = df.index
    cols = df.columns
    df = pd.DataFrame(np.round(df_imp), index=index, columns = cols)
    return df

def upsample(X, y, id_col, upsampler):
    print('Upsampling the minority class.')
    X_upsampled, y_upsampled = upsampler.fit_resample(X, y)
    cols = X.columns
    X = pd.DataFrame(X_upsampled, columns=cols, dtype=float)
    
    # Save the upsampled groups array
    upsampled_groups = X[id_col]

    return X, y_upsampled, upsampled_groups

def scale(X, scaler):
    print('Scaling input features.')
    
    ''' Perform Scaling
        Thank you for your guidance, @Miriam Farber
        https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
    '''
    X_scaled = scaler.fit_transform(X)
    index = X.index
    cols = X.columns
    X = pd.DataFrame(X_scaled, index=index, columns=cols)
    return X
