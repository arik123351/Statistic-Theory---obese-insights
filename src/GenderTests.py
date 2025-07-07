import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def test_gender_vs_obesity(df):
    """Test association between gender and obesity using Chi-square test"""
    try:
        # Check if columns exist
        if 'Gender' not in df.columns or 'is_obese' not in df.columns:
            print("Error: Required columns 'Gender' and/or 'is_obese' not found")
            return None, None

        # Remove any missing values
        df_clean = df[['Gender', 'is_obese']].dropna()

        if len(df_clean) == 0:
            print("Error: No valid data after removing missing values")
            return None, None

        contingency_table = pd.crosstab(df_clean['Gender'], df_clean['is_obese'])

        # Check if we have enough data for chi-square test
        if contingency_table.size < 4 or (contingency_table < 5).any().any():
            print("Warning: Some cells have counts < 5, chi-square test may not be reliable")

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print("Q1 - Gender vs Obesity")
        print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
        print(f"Contingency table:\n{contingency_table}\n")

        return chi2, p

    except Exception as e:
        print(f"Error in gender vs obesity test: {e}")
        return None, None


def test_family_history_vs_obesity(df):
    """Test association between family history and obesity using Chi-square test"""
    try:
        # Check if columns exist
        if 'family_history_with_overweight' not in df.columns or 'is_obese' not in df.columns:
            print("Error: Required columns 'family_history_with_overweight' and/or 'is_obese' not found")
            return None, None

        # Remove any missing values
        df_clean = df[['family_history_with_overweight', 'is_obese']].dropna()

        if len(df_clean) == 0:
            print("Error: No valid data after removing missing values")
            return None, None

        contingency_table = pd.crosstab(df_clean['family_history_with_overweight'], df_clean['is_obese'])

        # Check if we have enough data for chi-square test
        if contingency_table.size < 4 or (contingency_table < 5).any().any():
            print("Warning: Some cells have counts < 5, chi-square test may not be reliable")

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print("Q2 - Family History vs Obesity")
        print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
        print(f"Contingency table:\n{contingency_table}\n")

        return chi2, p

    except Exception as e:
        print(f"Error in family history vs obesity test: {e}")
        return None, None


def test_transport_vs_obesity(df):
    """Test association between transport modes and obesity using Chi-square test"""
    print("Q7 - Transport Mode vs Obesity")
    results = {}

    # First, check if we have a single transport column or multiple dummy columns
    transport_columns = [
        'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
        'MTRANS_Public_Transportation', 'MTRANS_Walking'
    ]

    # Check which transport columns exist
    existing_transport_cols = [col for col in transport_columns if col in df.columns]

    if not existing_transport_cols:
        # Maybe there's a single transport column
        transport_like_cols = [col for col in df.columns if 'transport' in col.lower() or 'mtrans' in col.lower()]
        if transport_like_cols:
            print(f"Found transport-related columns: {transport_like_cols}")
            existing_transport_cols = transport_like_cols
        else:
            print("No transport columns found in the dataset")
            return results

    if 'is_obese' not in df.columns:
        print("Error: 'is_obese' column not found")
        return results

    for col in existing_transport_cols:
        try:
            # Remove missing values
            df_clean = df[[col, 'is_obese']].dropna()

            if len(df_clean) == 0:
                print(f"{col}: No valid data after removing missing values")
                continue

            if df_clean[col].nunique() <= 1:  # Skip if no variation
                print(f"{col}: Not enough variation to test (only {df_clean[col].nunique()} unique values)")
                continue

            table = pd.crosstab(df_clean[col], df_clean['is_obese'])

            # Check if we have enough data for chi-square test
            if table.size < 4 or (table < 5).any().any():
                print(f"{col}: Warning - Some cells have counts < 5, chi-square test may not be reliable")

            chi2, p, dof, expected = chi2_contingency(table)
            print(f"{col}: chi2 = {chi2:.4f}, p = {p:.4f}")
            results[col] = (chi2, p)

        except Exception as e:
            print(f"Error testing {col}: {e}")

    return results


def logistic_regression_obesity(df):
    """Perform logistic regression to predict obesity"""
    print("Q10 - Logistic Regression on Obesity")

    # Check if target variable exists
    if 'is_obese' not in df.columns:
        print("Error: 'is_obese' column not found")
        return None

    # Define potential predictors
    potential_features = ['FAVC', 'FCVC', 'NCP', 'CAEC', 'FAF', 'CH2O',
                          'TUE', 'CALC', 'SMOKE', 'family_history_with_overweight']

    # Check which features exist in the dataset
    available_features = [col for col in potential_features if col in df.columns]

    if not available_features:
        print("Error: None of the expected predictor variables found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return None

    print(f"Using features: {available_features}")

    try:
        # Create a copy for modeling
        df_model = df[available_features + ['is_obese']].copy()

        # Remove rows with missing values
        df_model = df_model.dropna()

        if len(df_model) == 0:
            print("Error: No valid data after removing missing values")
            return None

        # Convert categorical variables to numeric
        encoders = {}
        for col in available_features:
            if df_model[col].dtype == 'object' or df_model[col].dtype.name == 'category':
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col].astype(str))
                encoders[col] = le

        # Prepare features and target
        X = df_model[available_features]
        y = df_model['is_obese']

        # Convert target to numeric if it's not already
        target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))

        # Fit logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        print("\nLogistic Regression Results:")
        print("=" * 50)
        print(f"Model Score (Accuracy): {model.score(X, y):.4f}")
        print(f"Number of features: {len(available_features)}")
        print(f"Sample size: {len(df_model)}")

        # Print coefficients
        print("\nCoefficients:")
        print("=" * 20)
        for i, feature in enumerate(available_features):
            coef = model.coef_[0][i]
            print(f"{feature}: {coef:.4f}")

        print(f"\nIntercept: {model.intercept_[0]:.4f}")

        # Print odds ratios for easier interpretation
        print("\nOdds Ratios:")
        print("=" * 20)
        for i, feature in enumerate(available_features):
            odds_ratio = np.exp(model.coef_[0][i])
            print(f"{feature}: {odds_ratio:.4f}")

        # Print classification report
        print("\nClassification Report:")
        print("=" * 30)
        if target_encoder:
            target_names = target_encoder.classes_
        else:
            target_names = ['Not Obese', 'Obese']

        print(classification_report(y, y_pred, target_names=target_names))

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("=" * 20)
        cm = confusion_matrix(y, y_pred)
        print(cm)

        return {
            'model': model,
            'feature_names': available_features,
            'encoders': encoders,
            'target_encoder': target_encoder,
            'accuracy': model.score(X, y),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    except Exception as e:
        print(f"Error in logistic regression: {e}")
        return None


def run_all_tests(df):
    """Run all obesity analysis tests"""
    print("Running Obesity Analysis Tests")
    print("=" * 50)

    # Test 1: Gender vs Obesity
    test_gender_vs_obesity(df)

    # Test 2: Family History vs Obesity
    test_family_history_vs_obesity(df)

    # Test 3: Transport vs Obesity
    test_transport_vs_obesity(df)

    # Test 4: Logistic Regression
    model = logistic_regression_obesity(df)

    return model

# Example usage:
# df = pd.read_csv('your_data.csv')
# model = run_all_tests(df)