"""
Run Chronos model with tuned train/val/test split
Compare against baseline
"""

import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# Configuration
# ============================================================================

# Use same splits as baseline tuning
TRAIN_CUTOFF = pd.Timestamp('2023-12-31')
VAL_CUTOFF = pd.Timestamp('2024-12-31')

# Chronos model
CHRONOS_MODEL = 'Chronos-Bolt'  # Options: 'Chronos-Bolt', 'Chronos'
PREDICTION_LENGTH = 30  # Forecast 30 time steps ahead

# ============================================================================
# Load Data
# ============================================================================

def load_and_split_data():
    """Load processed data and split into train/val/test"""
    print("Loading processed data...")

    train = pd.read_csv('data/processed/train.csv', parse_dates=['Stay_Date', 'as_of_date'])
    test = pd.read_csv('data/processed/test.csv', parse_dates=['Stay_Date', 'as_of_date'])

    full_data = pd.concat([train, test], ignore_index=True)

    # Split by stay dates
    train_data = full_data[full_data['Stay_Date'] <= TRAIN_CUTOFF].copy()
    val_data = full_data[(full_data['Stay_Date'] > TRAIN_CUTOFF) &
                         (full_data['Stay_Date'] <= VAL_CUTOFF)].copy()
    test_data = full_data[full_data['Stay_Date'] > VAL_CUTOFF].copy()

    print(f"\nTrain: {len(train_data):,} samples, stay dates up to {TRAIN_CUTOFF}")
    print(f"Val:   {len(val_data):,} samples, stay dates {TRAIN_CUTOFF} to {VAL_CUTOFF}")
    print(f"Test:  {len(test_data):,} samples, stay dates after {VAL_CUTOFF}")

    return train_data, val_data, test_data


# ============================================================================
# Prepare for AutoGluon TimeSeries
# ============================================================================

def prepare_timeseries_data(df, name='data'):
    """
    Convert to TimeSeriesDataFrame

    Note: For booking curves, each (Property_Code, Stay_Date) is a separate time series
    The challenge: test stay dates are unseen (cold start problem)
    Chronos foundation models should handle this better than traditional models
    """

    # AutoGluon needs sorted data
    df = df.sort_values(['Property_Code', 'Stay_Date', 'as_of_date']).reset_index(drop=True)

    # Create TimeSeriesDataFrame
    # Each (Property, StayDate) combination is one time series
    # Rename rooms_on_books to 'target' - AutoGluon expects this column name
    df_renamed = df.rename(columns={'rooms_on_books': 'target'})

    # Add item_id column (combination of Property_Code and Stay_Date)
    df_renamed['item_id'] = df_renamed['Property_Code'] + '_' + df_renamed['Stay_Date'].astype(str)

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df_renamed[['item_id', 'as_of_date', 'target']],
        id_column='item_id',
        timestamp_column='as_of_date'
    )

    print(f"\n{name} TimeSeriesDataFrame:")
    print(f"  Shape: {ts_df.shape}")
    print(f"  Num time series: {ts_df.num_items}")
    print(f"  Date range: {ts_df.index.get_level_values('timestamp').min()} to {ts_df.index.get_level_values('timestamp').max()}")

    return ts_df


# ============================================================================
# Train Chronos Model
# ============================================================================

def train_chronos(train_ts, val_ts=None, model_name='Chronos-Bolt'):
    """
    Train Chronos zero-shot model

    Note: "Training" here means loading the pre-trained model and
    optionally fine-tuning on your data. For pure zero-shot, no training needed.
    """

    print("\n" + "="*80)
    print(f"TRAINING CHRONOS MODEL: {model_name}")
    print("="*80)

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        path=f'outputs/models/chronos_{model_name.lower().replace("-", "_")}',
        eval_metric='MAPE',
        verbosity=2
    )

    print(f"\nUsing {model_name} (zero-shot foundation model)")
    print(f"Prediction length: {PREDICTION_LENGTH} time steps")
    print(f"This will download the pre-trained model on first run...")
    print()

    # Fit with Chronos
    # For zero-shot: just specify the model
    # For fine-tuning: can add time_limit
    predictor.fit(
        train_ts,
        hyperparameters={model_name: {}},
        time_limit=None,  # None = pure zero-shot, no fine-tuning
    )

    return predictor


# ============================================================================
# Evaluate
# ============================================================================

def evaluate_predictions(actual_df, predictions, split_name='Test'):
    """Calculate metrics"""

    # Convert predictions to DataFrame
    pred_df = predictions.reset_index()

    # Merge with actuals
    merged = actual_df.merge(
        pred_df,
        left_on=['Property_Code', 'Stay_Date', 'as_of_date'],
        right_on=['item_id', 'timestamp'],
        how='inner',
        suffixes=('_actual', '_pred')
    )

    if len(merged) == 0:
        print(f"⚠️  Warning: No matching predictions for {split_name} set!")
        return {}

    # Metrics
    mae = mean_absolute_error(merged['rooms_on_books'], merged['mean'])
    rmse = np.sqrt(mean_squared_error(merged['rooms_on_books'], merged['mean']))

    non_zero = merged['rooms_on_books'] > 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((merged.loc[non_zero, 'rooms_on_books'] -
                               merged.loc[non_zero, 'mean']) /
                              merged.loc[non_zero, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    print(f"\n📊 {split_name} Set Results:")
    print(f"  MAE:  {mae:.2f} rooms")
    print(f"  RMSE: {rmse:.2f} rooms")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Samples: {len(merged):,}")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'n_samples': len(merged)}, merged


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("CHRONOS ZERO-SHOT FORECASTING")
    print("With Train/Val/Test Split")
    print("="*80)

    # Load data
    train_df, val_df, test_df = load_and_split_data()

    # Prepare TimeSeriesDataFrames
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)

    train_ts = prepare_timeseries_data(train_df, 'Train')
    val_ts = prepare_timeseries_data(val_df, 'Validation')
    test_ts = prepare_timeseries_data(test_df, 'Test')

    # Train model
    predictor = train_chronos(train_ts, val_ts, model_name=CHRONOS_MODEL)

    # Validate
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)

    try:
        val_preds = predictor.predict(val_ts)
        val_metrics, val_merged = evaluate_predictions(val_df, val_preds, 'Validation')
    except Exception as e:
        print(f"⚠️  Validation prediction failed: {e}")
        val_metrics = {}

    # Test
    print("\n" + "="*80)
    print("TEST SET EVALUATION (FINAL)")
    print("="*80)

    try:
        test_preds = predictor.predict(test_ts)
        test_metrics, test_merged = evaluate_predictions(test_df, test_preds, 'Test')

        # Save results
        os.makedirs('outputs/predictions', exist_ok=True)
        os.makedirs('outputs/metrics', exist_ok=True)

        test_merged.to_csv('outputs/predictions/chronos_test_predictions.csv', index=False)

        metrics_df = pd.DataFrame([{
            'model': f'{CHRONOS_MODEL} (zero-shot)',
            'split': 'test',
            **test_metrics
        }])
        metrics_df.to_csv('outputs/metrics/chronos_test_metrics.csv', index=False)

    except Exception as e:
        print(f"⚠️  Test prediction failed: {e}")
        test_metrics = {}

    # Compare with baseline
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)

    baseline_metrics = pd.read_csv('outputs/metrics/baseline_final_metrics.csv')
    baseline_mae = baseline_metrics['MAE'].values[0]

    if test_metrics:
        chronos_mae = test_metrics['MAE']
        improvement = ((baseline_mae - chronos_mae) / baseline_mae) * 100

        print(f"Baseline (Additive Pickup):  {baseline_mae:.2f} rooms")
        print(f"Chronos ({CHRONOS_MODEL}):      {chronos_mae:.2f} rooms")
        print(f"Improvement: {improvement:.1f}%")

        if chronos_mae < baseline_mae:
            print("\n✅ Chronos beats the baseline!")
        else:
            print("\n⚠️  Chronos did not beat baseline (cold start problem?)")

    print("\n" + "="*80)
    print("✅ DONE")
    print("="*80)


if __name__ == '__main__':
    main()
