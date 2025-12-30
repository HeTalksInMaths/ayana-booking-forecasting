"""
Ayana Resort Booking Curve Forecasting with AutoGluon + Chronos Models
Author: AutoGluon Direct (No Assistant Needed)
Description: Use AutoGluon's zero-shot Chronos-2 and Chronos-Bolt foundation models

Chronos Models:
- Chronos-Bolt: Fast, efficient zero-shot forecasting (Amazon's latest)
- Chronos-2: Improved version of Chronos with better accuracy
- No training required - these are foundation models pre-trained on diverse time series
"""

import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# Configuration
# ============================================================================

# Data paths
DATA_PATH = '/Users/hetalksinmaths/Downloads/filtered_hotels.csv'
OUTPUT_DIR = './outputs'

# Model settings
PREDICTION_LENGTH = 30  # How many time steps to forecast
TEST_HORIZON_DAYS = 90  # How much data to hold out for testing

# Chronos model selection
# Options: 'chronos-bolt-small', 'chronos-bolt-base', 'chronos-bolt-large'
#          'chronos_tiny', 'chronos_mini', 'chronos_small', 'chronos_base', 'chronos_large'
CHRONOS_MODEL = 'chronos-bolt-base'  # Fast and accurate

# ============================================================================
# Data Loading (same as before)
# ============================================================================

def load_ayana_data(filepath):
    """Load Ayana booking curve data and transform to long format"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"Original shape: {df.shape}")

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
# AutoGluon with Chronos Models (Zero-Shot)
# ============================================================================

def train_chronos_model(train_df, prediction_length=30, model_name='chronos-bolt-base'):
    """
    Train AutoGluon with Chronos zero-shot foundation models

    Chronos models are pre-trained on diverse time series datasets
    and can forecast without additional training (zero-shot).

    Args:
        train_df: Training data
        prediction_length: Number of time steps to forecast
        model_name: Which Chronos model to use
                   - 'chronos-bolt-small': Fastest, 50M params
                   - 'chronos-bolt-base': Balanced (RECOMMENDED), 200M params
                   - 'chronos-bolt-large': Most accurate, 600M params
                   - 'chronos_tiny': Legacy, 8M params
                   - 'chronos_small': Legacy, 20M params
                   - 'chronos_base': Legacy, 200M params
    """

    print(f"\n{'='*80}")
    print(f"Using Chronos Model: {model_name}")
    print(f"{'='*80}")
    print(f"Prediction length: {prediction_length} time steps")

    # Create TimeSeriesDataFrame
    print("\nCreating TimeSeriesDataFrame...")
    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df.rename(columns={'rooms_on_books': 'target'}),
        id_column=['Property_Code', 'Stay_Date'],
        timestamp_column='as_of_date',
        target='target'
    )

    print(f"TimeSeriesDataFrame shape: {train_ts.shape}")
    print(f"Number of time series: {train_ts.num_items}")

    # Initialize predictor with Chronos
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=f'{OUTPUT_DIR}/models/chronos_{model_name}',
        eval_metric='MAPE',
        verbosity=2
    )

    # Fit with Chronos - zero-shot mode
    print("\nFitting Chronos model (zero-shot, no training required)...")
    print("This downloads the pre-trained model on first run...\n")

    predictor.fit(
        train_ts,
        hyperparameters={
            model_name: {}  # Just specify the model, no hyperparameters needed
        },
        time_limit=None,  # No time limit needed for zero-shot
    )

    return predictor


def train_chronos_ensemble(train_df, prediction_length=30):
    """
    Train ensemble of multiple Chronos models for best accuracy

    Combines:
    - Chronos-Bolt (fast)
    - Chronos-2 (accurate)
    - Traditional models (ETS, AutoARIMA)
    """

    print(f"\n{'='*80}")
    print("Training Chronos Ensemble")
    print(f"{'='*80}")

    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df.rename(columns={'rooms_on_books': 'target'}),
        id_column=['Property_Code', 'Stay_Date'],
        timestamp_column='as_of_date',
        target='target'
    )

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=f'{OUTPUT_DIR}/models/chronos_ensemble',
        eval_metric='MAPE',
        verbosity=2
    )

    # Ensemble of Chronos models + traditional methods
    print("\nTraining ensemble of:")
    print("  - Chronos-Bolt-Base (zero-shot)")
    print("  - Chronos-Base (zero-shot)")
    print("  - ETS (traditional)")
    print("  - AutoARIMA (traditional)")
    print()

    predictor.fit(
        train_ts,
        hyperparameters={
            'Chronos-Bolt': {},  # Latest fast model
            'Chronos': {},       # Accurate foundation model
            'ETS': {},           # Exponential smoothing
            'AutoARIMA': {},     # ARIMA
        },
        time_limit=3600,  # 1 hour for ensemble
    )

    return predictor


# ============================================================================
# Evaluation
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
# Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("🏨 Ayana Booking Curve Forecasting")
    print("   Using AutoGluon Chronos Zero-Shot Models")
    print("=" * 80)
    print()

    # Create output directories
    os.makedirs(f'{OUTPUT_DIR}/predictions', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/metrics', exist_ok=True)

    # Step 1: Load data
    print("📂 Step 1: Loading data")
    print("-" * 80)
    df = load_ayana_data(DATA_PATH)
    print(f"\nLoaded {len(df):,} booking curve snapshots")
    print(f"Properties: {df['Property_Code'].unique().tolist()}")

    # Step 2: Split data
    print("\n" + "=" * 80)
    print("✂️  Step 2: Train/test split")
    print("-" * 80)
    train_df, test_df = create_train_test_split(df, test_horizon_days=TEST_HORIZON_DAYS)

    # Save processed data
    train_df.to_csv('./data/processed/train.csv', index=False)
    test_df.to_csv('./data/processed/test.csv', index=False)

    # Step 3: Choose your approach
    print("\n" + "=" * 80)
    print("🤖 Step 3: Model Selection")
    print("-" * 80)
    print("Choose one:")
    print("  1. Single Chronos model (fastest, zero-shot)")
    print("  2. Chronos ensemble (best accuracy)")
    print()

    # Option 1: Single Chronos model (RECOMMENDED FOR SPEED)
    print("Running Option 1: Single Chronos-Bolt model...")
    predictor = train_chronos_model(
        train_df,
        prediction_length=PREDICTION_LENGTH,
        model_name=CHRONOS_MODEL
    )

    # Option 2: Ensemble (uncomment to use instead)
    # print("Running Option 2: Chronos ensemble...")
    # predictor = train_chronos_ensemble(
    #     train_df,
    #     prediction_length=PREDICTION_LENGTH
    # )

    # Step 4: Predict
    print("\n" + "=" * 80)
    print("🔮 Step 4: Generating predictions")
    print("-" * 80)

    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df.rename(columns={'rooms_on_books': 'target'}),
        id_column=['Property_Code', 'Stay_Date'],
        timestamp_column='as_of_date',
        target='target'
    )

    predictions = predictor.predict(test_ts)
    predictions_df = predictions.reset_index()

    # Step 5: Evaluate
    print("\n" + "=" * 80)
    print("📈 Step 5: Evaluation")
    print("-" * 80)
    metrics = evaluate_predictions(test_df, predictions_df)

    # Step 6: Model info
    print("\n" + "=" * 80)
    print("🏆 Model Information")
    print("-" * 80)
    try:
        leaderboard = predictor.leaderboard(test_ts, silent=True)
        print(leaderboard.to_string())
    except Exception as e:
        print(f"Could not generate leaderboard: {e}")

    # Step 7: Save
    print("\n" + "=" * 80)
    print("💾 Saving results")
    print("-" * 80)

    predictions_df.to_csv(f'{OUTPUT_DIR}/predictions/chronos_predictions.csv', index=False)
    print(f"✓ Predictions: {OUTPUT_DIR}/predictions/chronos_predictions.csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = CHRONOS_MODEL
    metrics_df.to_csv(f'{OUTPUT_DIR}/metrics/chronos_metrics.csv', index=False)
    print(f"✓ Metrics: {OUTPUT_DIR}/metrics/chronos_metrics.csv")

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)
    print(f"\nModel: {CHRONOS_MODEL}")
    print(f"MAE: {metrics.get('MAE', 0):.2f} rooms")
    print(f"\nZero-shot inference - no training required!")

if __name__ == '__main__':
    main()
