"""
Ayana Resort Booking Curve Forecasting with AutoGluon
Author: AutoGluon Assistant
Description: Train AutoGluon TimeSeries models on Ayana booking curve data
"""

import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# 1. Load and Transform Data from Wide to Long Format
# ============================================================================

def load_ayana_data(filepath):
    """Load Ayana booking curve data and transform to long format"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

    # Identify booking curve columns (Dmin and Dplus)
    d_cols = [col for col in df.columns if col.startswith('Dmin') or col.startswith('Dplus')]
    print(f"Found {len(d_cols)} booking curve columns")

    # Metadata columns
    id_cols = ['Stay_Date', 'Property_Code', 'Day_of_Week']

    # Melt from wide to long format
    print("Transforming from wide to long format...")
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name='snapshot_type',
        value_name='rooms_on_books'
    )

    # Extract days_to_arrival from column names
    # Dmin100 → -100, Dmin001 → -1, Dplus000 → 0, Dplus001 → 1
    def parse_days(col_name):
        if col_name.startswith('Dmin'):
            return -int(col_name.replace('Dmin', ''))
        elif col_name.startswith('Dplus'):
            return int(col_name.replace('Dplus', ''))
        return None

    long_df['days_to_arrival'] = long_df['snapshot_type'].apply(parse_days)

    # Clean data
    long_df = long_df.dropna(subset=['rooms_on_books'])
    long_df['Stay_Date'] = pd.to_datetime(long_df['Stay_Date'])

    # Create as_of_date (snapshot date) = stay_date - days_to_arrival
    long_df['as_of_date'] = long_df['Stay_Date'] - pd.to_timedelta(-long_df['days_to_arrival'], unit='D')

    # Sort
    long_df = long_df.sort_values(['Property_Code', 'Stay_Date', 'as_of_date']).reset_index(drop=True)

    result = long_df[['Property_Code', 'Stay_Date', 'as_of_date', 'days_to_arrival', 'rooms_on_books', 'Day_of_Week']]

    print(f"Transformed shape: {result.shape}")
    return result


# ============================================================================
# 2. Train/Test Split
# ============================================================================

def create_train_test_split(df, test_horizon_days=90):
    """Split data into train and test sets"""
    cutoff_date = df['Stay_Date'].max() - pd.Timedelta(days=test_horizon_days)

    train = df[df['Stay_Date'] <= cutoff_date].copy()
    test = df[df['Stay_Date'] > cutoff_date].copy()

    print(f"\nTrain: {train['Stay_Date'].min()} to {train['Stay_Date'].max()}")
    print(f"Test:  {test['Stay_Date'].min()} to {test['Stay_Date'].max()}")
    print(f"Train samples: {len(train):,}")
    print(f"Test samples:  {len(test):,}")

    return train, test


# ============================================================================
# 3. AutoGluon TimeSeries Forecasting
# ============================================================================

def train_autogluon_model(train_df, prediction_length=30, time_limit=1800):
    """Train AutoGluon TimeSeries model

    Args:
        train_df: Training data
        prediction_length: Number of time steps to forecast
        time_limit: Training time limit in seconds
    """

    # Create TimeSeriesDataFrame
    # AutoGluon expects: item_id, timestamp, target
    print("\nCreating TimeSeriesDataFrame...")
    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df.rename(columns={'rooms_on_books': 'target'}),
        id_column=['Property_Code', 'Stay_Date'],  # Group by property and stay date
        timestamp_column='as_of_date',
        target='target'
    )

    print(f"TimeSeriesDataFrame shape: {train_ts.shape}")
    print(f"Number of time series: {train_ts.num_items}")

    # Initialize predictor
    print(f"\nInitializing AutoGluon TimeSeriesPredictor...")
    print(f"Prediction length: {prediction_length} time steps")
    print(f"Time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path='./outputs/models/ag_models',
        eval_metric='MAPE',  # Mean Absolute Percentage Error
        verbosity=2
    )

    # Train with presets
    # Options: 'fast_training', 'medium_quality', 'high_quality', 'best_quality'
    print("\nStarting model training...")
    predictor.fit(
        train_ts,
        presets='medium_quality',
        time_limit=time_limit,
        num_val_windows=3,  # Cross-validation windows
    )

    return predictor


# ============================================================================
# 4. Evaluation
# ============================================================================

def evaluate_predictions(test_df, predictions_df):
    """Calculate evaluation metrics"""

    # Merge predictions with actuals
    merged = test_df.merge(
        predictions_df,
        left_on=['Property_Code', 'Stay_Date', 'as_of_date'],
        right_on=['item_id', 'timestamp'],
        how='inner',
        suffixes=('_actual', '_pred')
    )

    if len(merged) == 0:
        print("⚠️  Warning: No matching predictions found!")
        return {}

    # Calculate metrics
    mae = mean_absolute_error(merged['rooms_on_books'], merged['mean'])
    rmse = np.sqrt(mean_squared_error(merged['rooms_on_books'], merged['mean']))

    # Avoid division by zero in MAPE
    non_zero_mask = merged['rooms_on_books'] != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((merged.loc[non_zero_mask, 'rooms_on_books'] -
                               merged.loc[non_zero_mask, 'mean']) /
                              merged.loc[non_zero_mask, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    print(f"\n📊 Evaluation Metrics:")
    print(f"  MAE:  {mae:.2f} rooms")
    print(f"  RMSE: {rmse:.2f} rooms")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Samples evaluated: {len(merged):,}")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'n_samples': len(merged)}


# ============================================================================
# 5. Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("🏨 Ayana Resort Booking Curve Forecasting with AutoGluon")
    print("=" * 80)
    print()

    # Configuration
    DATA_PATH = '/Users/hetalksinmaths/Downloads/filtered_hotels.csv'
    TEST_HORIZON_DAYS = 90
    PREDICTION_LENGTH = 30  # Forecast 30 time steps ahead
    TIME_LIMIT = 1800  # 30 minutes

    # Create output directories
    os.makedirs('./outputs/predictions', exist_ok=True)
    os.makedirs('./outputs/models', exist_ok=True)
    os.makedirs('./outputs/metrics', exist_ok=True)

    # Step 1: Load data
    print("📂 Step 1: Loading and transforming data")
    print("-" * 80)
    df = load_ayana_data(DATA_PATH)
    print(f"\nLoaded {len(df):,} booking curve snapshots")
    print(f"Properties: {df['Property_Code'].unique().tolist()}")
    print(f"Date range: {df['Stay_Date'].min()} to {df['Stay_Date'].max()}")

    # Step 2: Split data
    print("\n" + "=" * 80)
    print("✂️  Step 2: Splitting train/test")
    print("-" * 80)
    train_df, test_df = create_train_test_split(df, test_horizon_days=TEST_HORIZON_DAYS)

    # Save processed data
    train_df.to_csv('./data/processed/train.csv', index=False)
    test_df.to_csv('./data/processed/test.csv', index=False)
    print("\n✓ Saved processed data to ./data/processed/")

    # Step 3: Train AutoGluon model
    print("\n" + "=" * 80)
    print("🤖 Step 3: Training AutoGluon TimeSeries model")
    print("-" * 80)
    print("This may take 30-60 minutes depending on your machine.")
    print()

    predictor = train_autogluon_model(
        train_df,
        prediction_length=PREDICTION_LENGTH,
        time_limit=TIME_LIMIT
    )

    # Step 4: Make predictions
    print("\n" + "=" * 80)
    print("🔮 Step 4: Generating predictions on test set")
    print("-" * 80)

    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df.rename(columns={'rooms_on_books': 'target'}),
        id_column=['Property_Code', 'Stay_Date'],
        timestamp_column='as_of_date',
        target='target'
    )

    predictions = predictor.predict(test_ts)
    predictions_df = predictions.reset_index()

    print(f"\n✓ Generated {len(predictions_df):,} predictions")

    # Step 5: Evaluate
    print("\n" + "=" * 80)
    print("📈 Step 5: Evaluating model performance")
    print("-" * 80)
    metrics = evaluate_predictions(test_df, predictions_df)

    # Step 6: Model leaderboard
    print("\n" + "=" * 80)
    print("🏆 Step 6: Model Leaderboard")
    print("-" * 80)
    try:
        leaderboard = predictor.leaderboard(test_ts, silent=True)
        print(leaderboard.to_string())
    except Exception as e:
        print(f"Could not generate leaderboard: {e}")

    # Step 7: Save results
    print("\n" + "=" * 80)
    print("💾 Step 7: Saving results")
    print("-" * 80)

    predictions_df.to_csv('./outputs/predictions/autogluon_predictions.csv', index=False)
    print("✓ Predictions saved to: ./outputs/predictions/autogluon_predictions.csv")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('./outputs/metrics/evaluation_metrics.csv', index=False)
    print("✓ Metrics saved to: ./outputs/metrics/evaluation_metrics.csv")

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
