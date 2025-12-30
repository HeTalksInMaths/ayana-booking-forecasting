"""
Baseline models for booking curve forecasting
Author: AutoGluon Assistant
Description: Simple baseline models to compare against AutoGluon
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def additive_pickup_forecast(train_df, test_df, verbose=True):
    """
    Additive pickup baseline model

    Forecasts remaining rooms to be booked by averaging historical pickup curves.

    Logic:
    1. Calculate "final occupancy" (rooms on books at day 0)
    2. Calculate "remaining pickup" = final_occ - current_rooms
    3. Average remaining pickup by property and days_to_arrival
    4. Forecast test set by adding average remaining pickup to current rooms

    Args:
        train_df: Training data with columns [Property_Code, Stay_Date, as_of_date,
                  days_to_arrival, rooms_on_books]
        test_df: Test data with same structure
        verbose: Print detailed information

    Returns:
        test_enriched: Test data with forecasts
        mae: Mean Absolute Error
    """

    if verbose:
        print("\n🔧 Additive Pickup Baseline Model")
        print("-" * 60)

    # Get final occupancy (last snapshot for each stay date)
    finals = (train_df.sort_values('as_of_date')
                      .groupby(['Property_Code', 'Stay_Date'])['rooms_on_books']
                      .last()
                      .rename('final_occ')
                      .reset_index())

    if verbose:
        print(f"Calculated final occupancy for {len(finals):,} stay dates")

    # Merge with full dataset
    train_enriched = train_df.merge(finals, on=['Property_Code', 'Stay_Date'], how='left')

    # Calculate remaining pickup
    train_enriched['remaining'] = train_enriched['final_occ'] - train_enriched['rooms_on_books']

    # Average remaining pickup by property and days_to_arrival
    baseline_pickup = (train_enriched.groupby(['Property_Code', 'days_to_arrival'])['remaining']
                                     .mean()
                                     .rename('avg_remaining')
                                     .reset_index())

    if verbose:
        print(f"Computed average pickup curves for {len(baseline_pickup):,} property-day combinations")

    # Apply to test set
    test_enriched = test_df.merge(baseline_pickup, on=['Property_Code', 'days_to_arrival'], how='left')
    test_enriched['avg_remaining'] = test_enriched['avg_remaining'].fillna(0)

    # Forecast = current_rooms + avg_remaining
    test_enriched['forecast'] = test_enriched['rooms_on_books'] + test_enriched['avg_remaining']

    # Ensure non-negative forecasts
    test_enriched['forecast'] = test_enriched['forecast'].clip(lower=0)

    # Evaluate
    mae = mean_absolute_error(test_enriched['rooms_on_books'], test_enriched['forecast'])
    rmse = np.sqrt(mean_squared_error(test_enriched['rooms_on_books'], test_enriched['forecast']))

    # MAPE
    non_zero_mask = test_enriched['rooms_on_books'] != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((test_enriched.loc[non_zero_mask, 'rooms_on_books'] -
                               test_enriched.loc[non_zero_mask, 'forecast']) /
                              test_enriched.loc[non_zero_mask, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    if verbose:
        print(f"\n📊 Results:")
        print(f"  MAE:  {mae:.2f} rooms")
        print(f"  RMSE: {rmse:.2f} rooms")
        print(f"  MAPE: {mape:.2f}%")

    return test_enriched, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def naive_forecast(train_df, test_df, verbose=True):
    """
    Naive baseline: Use last available value as forecast

    For each test observation, use the most recent historical value
    at the same days_to_arrival for that property.

    Args:
        train_df: Training data
        test_df: Test data
        verbose: Print information

    Returns:
        test_enriched: Test data with forecasts
        metrics: Dictionary of evaluation metrics
    """

    if verbose:
        print("\n🔧 Naive Baseline Model (Last Value)")
        print("-" * 60)

    # Get last value for each property and days_to_arrival
    last_values = (train_df.sort_values('as_of_date')
                           .groupby(['Property_Code', 'days_to_arrival'])['rooms_on_books']
                           .last()
                           .rename('forecast')
                           .reset_index())

    # Apply to test set
    test_enriched = test_df.merge(last_values, on=['Property_Code', 'days_to_arrival'], how='left')

    # Fill missing with global mean
    global_mean = train_df['rooms_on_books'].mean()
    test_enriched['forecast'] = test_enriched['forecast'].fillna(global_mean)

    # Evaluate
    mae = mean_absolute_error(test_enriched['rooms_on_books'], test_enriched['forecast'])
    rmse = np.sqrt(mean_squared_error(test_enriched['rooms_on_books'], test_enriched['forecast']))

    non_zero_mask = test_enriched['rooms_on_books'] != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((test_enriched.loc[non_zero_mask, 'rooms_on_books'] -
                               test_enriched.loc[non_zero_mask, 'forecast']) /
                              test_enriched.loc[non_zero_mask, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    if verbose:
        print(f"\n📊 Results:")
        print(f"  MAE:  {mae:.2f} rooms")
        print(f"  RMSE: {rmse:.2f} rooms")
        print(f"  MAPE: {mape:.2f}%")

    return test_enriched, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def moving_average_forecast(train_df, test_df, window=7, verbose=True):
    """
    Moving average baseline

    Forecasts using the moving average of recent observations
    at the same days_to_arrival for that property.

    Args:
        train_df: Training data
        test_df: Test data
        window: Number of recent observations to average
        verbose: Print information

    Returns:
        test_enriched: Test data with forecasts
        metrics: Dictionary of evaluation metrics
    """

    if verbose:
        print(f"\n🔧 Moving Average Baseline (window={window})")
        print("-" * 60)

    # Calculate moving average for each property and days_to_arrival
    train_sorted = train_df.sort_values(['Property_Code', 'days_to_arrival', 'as_of_date'])
    train_sorted['ma_forecast'] = (train_sorted.groupby(['Property_Code', 'days_to_arrival'])['rooms_on_books']
                                               .transform(lambda x: x.rolling(window, min_periods=1).mean()))

    # Get last MA value for each group
    ma_values = (train_sorted.groupby(['Property_Code', 'days_to_arrival'])['ma_forecast']
                             .last()
                             .rename('forecast')
                             .reset_index())

    # Apply to test set
    test_enriched = test_df.merge(ma_values, on=['Property_Code', 'days_to_arrival'], how='left')

    # Fill missing with global mean
    global_mean = train_df['rooms_on_books'].mean()
    test_enriched['forecast'] = test_enriched['forecast'].fillna(global_mean)

    # Evaluate
    mae = mean_absolute_error(test_enriched['rooms_on_books'], test_enriched['forecast'])
    rmse = np.sqrt(mean_squared_error(test_enriched['rooms_on_books'], test_enriched['forecast']))

    non_zero_mask = test_enriched['rooms_on_books'] != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((test_enriched.loc[non_zero_mask, 'rooms_on_books'] -
                               test_enriched.loc[non_zero_mask, 'forecast']) /
                              test_enriched.loc[non_zero_mask, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    if verbose:
        print(f"\n📊 Results:")
        print(f"  MAE:  {mae:.2f} rooms")
        print(f"  RMSE: {rmse:.2f} rooms")
        print(f"  MAPE: {mape:.2f}%")

    return test_enriched, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def run_all_baselines(train_df, test_df):
    """
    Run all baseline models and compare results

    Args:
        train_df: Training data
        test_df: Test data

    Returns:
        results_df: DataFrame with comparison of all models
    """

    print("\n" + "=" * 80)
    print("🔍 Running All Baseline Models")
    print("=" * 80)

    results = []

    # 1. Additive Pickup
    _, metrics = additive_pickup_forecast(train_df, test_df)
    results.append({'Model': 'Additive Pickup', **metrics})

    # 2. Naive
    _, metrics = naive_forecast(train_df, test_df)
    results.append({'Model': 'Naive (Last Value)', **metrics})

    # 3. Moving Average (different windows)
    for window in [3, 7, 14]:
        _, metrics = moving_average_forecast(train_df, test_df, window=window)
        results.append({'Model': f'Moving Average (w={window})', **metrics})

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')

    print("\n" + "=" * 80)
    print("📊 Baseline Model Comparison")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n✓ Best model: " + results_df.iloc[0]['Model'])

    return results_df


if __name__ == '__main__':
    # Example usage
    print("Loading data...")

    # Load processed data if it exists
    try:
        train_df = pd.read_csv('./data/processed/train.csv')
        test_df = pd.read_csv('./data/processed/test.csv')

        # Convert date columns
        train_df['Stay_Date'] = pd.to_datetime(train_df['Stay_Date'])
        train_df['as_of_date'] = pd.to_datetime(train_df['as_of_date'])
        test_df['Stay_Date'] = pd.to_datetime(test_df['Stay_Date'])
        test_df['as_of_date'] = pd.to_datetime(test_df['as_of_date'])

        print(f"Loaded {len(train_df):,} training samples and {len(test_df):,} test samples")

        # Run all baselines
        results = run_all_baselines(train_df, test_df)

        # Save results
        results.to_csv('./outputs/metrics/baseline_comparison.csv', index=False)
        print(f"\n💾 Results saved to: ./outputs/metrics/baseline_comparison.csv")

    except FileNotFoundError:
        print("⚠️  Error: Processed data not found.")
        print("Please run 'python run_autogluon.py' first to process the data.")
