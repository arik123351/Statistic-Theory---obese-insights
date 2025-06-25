import pandas as pd


def CleanData():
    df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    binary_map = {
        "Gender": {"Female": 0, "Male": 1},
        "family_history_with_overweight": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
    }

    df.replace(binary_map, inplace=True)

    # Ordinal encoding for CAEC and CALC
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["CAEC"] = df["CAEC"].map(caec_map)
    df["CALC"] = df["CALC"].map(calc_map)

    # One-hot encode 'MTRANS' and 'NObeyesdad'
    df = pd.get_dummies(df, columns=["MTRANS", "NObeyesdad"], drop_first=False)

    obese_mask = df.filter(like="NObeyesdad_").columns.str.contains("Obesity", case=False)
    obese_colnames = df.filter(like="NObeyesdad_").columns[obese_mask]
    df["is_obese"] = (df[obese_colnames].sum(axis=1) > 0).astype(int)

    return df