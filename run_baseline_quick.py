"""
Quick baseline evaluation for Ayana booking curves
Runs additive pickup baseline using historical stay dates
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ============================================================================
# 1. Load and Transform Data
# ============================================================================

def load_ayana_data(filepath):
    """Load and transform Ayana booking curve data"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")

    # Get booking curve columns
    d_cols = [col for col in df.columns if col.startswith('Dmin') or col.startswith('Dplus')]
    print(f"Found {len(d_cols)} booking curve columns")

    # Melt to long format
    id_cols = ['Stay_Date', 'Property_Code', 'Day_of_Week']
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name='snapshot_type',
        value_name='rooms_on_books'
    )

    # Parse days_to_arrival
    def parse_days(col_name):
        if col_name.startswith('Dmin'):
            return -int(col_name.replace('Dmin', ''))
        elif col_name.startswith('Dplus'):
            return int(col_name.replace('Dplus', ''))
        return None

    long_df['days_to_arrival'] = long_df['snapshot_type'].apply(parse_days)
    long_df = long_df.dropna(subset=['rooms_on_books'])
    long_df['Stay_Date'] = pd.to_datetime(long_df['Stay_Date'])
    long_df['as_of_date'] = long_df['Stay_Date'] - pd.to_timedelta(-long_df['days_to_arrival'], unit='D')

    result = long_df[['Property_Code', 'Stay_Date', 'as_of_date', 'days_to_arrival', 'rooms_on_books']].copy()
    result = result.sort_values(['Property_Code', 'Stay_Date', 'as_of_date']).reset_index(drop=True)

    print(f"Transformed to long format: {result.shape}")
    return result


def create_train_test_split(df, test_horizon_days=90):
    """Split by stay date (not as_of_date!)"""
    cutoff_date = df['Stay_Date'].max() - pd.Timedelta(days=test_horizon_days)

    train = df[df['Stay_Date'] <= cutoff_date].copy()
    test = df[df['Stay_Date'] > cutoff_date].copy()

    print(f"\nTrain stay dates: {train['Stay_Date'].min()} to {train['Stay_Date'].max()}")
    print(f"Test stay dates:  {test['Stay_Date'].min()} to {test['Stay_Date'].max()}")
    print(f"Train samples: {len(train):,}")
    print(f"Test samples:  {len(test):,}")

    return train, test


# ============================================================================
# 2. Additive Pickup Baseline
# ============================================================================

def additive_pickup_baseline(train_df, test_df):
    """
    Additive pickup using historical stay dates

    For each (property, days_to_arrival), calculate:
    1. Average final occupancy from historical stay dates
    2. Average current rooms from historical stay dates
    3. Average remaining pickup = avg_final - avg_current
    4. Forecast = current_rooms + avg_remaining_pickup
    """

    print("\n" + "="*80)
    print("ADDITIVE PICKUP BASELINE")
    print("="*80)

    # Get final occupancy for each stay date in training set
    finals = (train_df.sort_values('as_of_date')
                     .groupby(['Property_Code', 'Stay_Date'])['rooms_on_books']
                     .last()
                     .rename('final_occ')
                     .reset_index())

    print(f"Calculated final occupancy for {len(finals):,} historical stay dates")

    # Merge finals back
    train_enriched = train_df.merge(finals, on=['Property_Code', 'Stay_Date'], how='left')

    # Calculate remaining pickup
    train_enriched['remaining'] = train_enriched['final_occ'] - train_enriched['rooms_on_books']

    # Average remaining pickup by (property, days_to_arrival)
    baseline_pickup = (train_enriched.groupby(['Property_Code', 'days_to_arrival'])['remaining']
                                    .mean()
                                    .rename('avg_remaining')
                                    .reset_index())

    print(f"Computed average pickup for {len(baseline_pickup):,} (property, days_to_arrival) combinations")

    # Apply to test set
    test_forecast = test_df.merge(baseline_pickup, on=['Property_Code', 'days_to_arrival'], how='left')
    test_forecast['avg_remaining'] = test_forecast['avg_remaining'].fillna(0)
    test_forecast['forecast'] = test_forecast['rooms_on_books'] + test_forecast['avg_remaining']
    test_forecast['forecast'] = test_forecast['forecast'].clip(lower=0)

    # Calculate metrics
    mae = mean_absolute_error(test_forecast['rooms_on_books'], test_forecast['forecast'])
    rmse = np.sqrt(mean_squared_error(test_forecast['rooms_on_books'], test_forecast['forecast']))

    # MAPE (avoid division by zero)
    non_zero = test_forecast['rooms_on_books'] > 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((test_forecast.loc[non_zero, 'rooms_on_books'] -
                               test_forecast.loc[non_zero, 'forecast']) /
                              test_forecast.loc[non_zero, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    print(f"\n📊 RESULTS:")
    print(f"  MAE:  {mae:.2f} rooms")
    print(f"  RMSE: {rmse:.2f} rooms")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Samples evaluated: {len(test_forecast):,}")

    # Save results
    os.makedirs('outputs/predictions', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    test_forecast.to_csv('outputs/predictions/baseline_additive_pickup.csv', index=False)

    metrics_df = pd.DataFrame([{
        'model': 'Additive Pickup Baseline',
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'n_samples': len(test_forecast)
    }])
    metrics_df.to_csv('outputs/metrics/baseline_metrics.csv', index=False)

    return test_forecast, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ============================================================================
# 3. Main Execution
# ============================================================================

def main():
    print("="*80)
    print("AYANA BOOKING CURVE FORECASTING - BASELINE MODEL")
    print("="*80)
    print()

    DATA_PATH = '/Users/hetalksinmaths/Downloads/filtered_hotels.csv'

    # Step 1: Load data
    print("Step 1: Loading and transforming data")
    print("-"*80)
    df = load_ayana_data(DATA_PATH)

    # Step 2: Split
    print("\nStep 2: Creating train/test split")
    print("-"*80)
    train_df, test_df = create_train_test_split(df, test_horizon_days=90)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print("✓ Saved processed data to data/processed/")

    # Step 3: Run baseline
    print("\nStep 3: Running additive pickup baseline")
    print("-"*80)
    test_forecast, metrics = additive_pickup_baseline(train_df, test_df)

    print("\n" + "="*80)
    print("✅ BASELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved:")
    print(f"  - Predictions: outputs/predictions/baseline_additive_pickup.csv")
    print(f"  - Metrics: outputs/metrics/baseline_metrics.csv")
    print(f"  - Processed data: data/processed/train.csv, test.csv")
    print(f"\n🎯 Baseline MAE: {metrics['MAE']:.2f} rooms")
    print(f"   This is what we need to beat with Chronos!")


if __name__ == '__main__':
    main()
