
import pandas as pd


def create_df_cat():
    return pd.DataFrame({
        "color": ["red", "blue", "red"]
    })

def create_df_cat2():
    return pd.DataFrame({
        "gender": ["female", "male", "male"],
        "color": ["red", "blue", "red"]
    })

def create_df_float():
    return pd.DataFrame({
        "price": [1.5, 2.5, 3.5]
    })

def create_df_int():
    return pd.DataFrame({
        "amount": [1, 2, 3]
    })

def create_df_all():
    return pd.concat([
        create_df_cat(),
        create_df_float(),
        create_df_int()
    ], axis=1)
