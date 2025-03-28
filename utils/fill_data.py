import pandas as pd

def forward_fill_imputation(df, columns_to_fill=None, axis=1):
    """Forward fill imputation for DataFrame."""
    df_filled = df.copy(deep=True)
    
    if columns_to_fill is not None:
        # Apply forward fill to specified columns only
        df_filled[columns_to_fill] = df_filled[columns_to_fill].ffill(axis=axis)
    else:
        # Apply forward fill to entire DataFrame
        df_filled = df_filled.ffill(axis=axis)
    
    return df_filled


def backward_fill_imputation(df, columns_to_fill=None, axis=1):
    """Backward fill imputation for DataFrame."""
    df_filled = df.copy(deep=True)
    
    if columns_to_fill is not None:
        # Apply backward fill to specified columns only
        df_filled[columns_to_fill] = df_filled[columns_to_fill].bfill(axis=axis)
    else:
        # Apply backward fill to entire DataFrame
        df_filled = df_filled.bfill(axis=axis)
    
    return df_filled


def mixed_directional_fill(df, columns_to_fill=None, fill_method='ffill_then_bfill', axis=1):
    """Combined forward/backward fill imputation."""
    df_filled = df.copy(deep=True)
    
    if fill_method == 'ffill_then_bfill':
        df_filled = forward_fill_imputation(df_filled, columns_to_fill, axis)
        df_filled = backward_fill_imputation(df_filled, columns_to_fill, axis)
    elif fill_method == 'bfill_then_ffill':
        df_filled = backward_fill_imputation(df_filled, columns_to_fill, axis)
        df_filled = forward_fill_imputation(df_filled, columns_to_fill, axis)
    else:
        raise ValueError("fill_method must be either 'ffill_then_bfill' or 'bfill_then_ffill'")
    
    return df_filled


def linear_interpolation(df, columns_to_fill=None, axis=1):
    """Linear interpolation with edge filling."""
    df_filled = df.copy(deep=True)
    
    if columns_to_fill is not None:
        # Apply interpolation to specified columns
        df_filled[columns_to_fill] = df_filled[columns_to_fill].interpolate(method='linear', axis=axis)
        # Fill remaining NaNs at edges
        df_filled[columns_to_fill] = df_filled[columns_to_fill].ffill(axis=axis).bfill(axis=axis)
    else:
        # Apply interpolation to entire DataFrame
        df_filled = df_filled.interpolate(method='linear', axis=axis)
        df_filled = df_filled.ffill(axis=axis).bfill(axis=axis)
    
    return df_filled