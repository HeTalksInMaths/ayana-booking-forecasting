"""
Tune baseline model lookback window on validation set
Try different historical year combinations to find best hyperparameter
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import timedelta

# ============================================================================
# Load processed data
# ============================================================================

def load_processed_data():
    """Load the already processed long-format data"""
    print("Loading processed data...")
    train = pd.read_csv('data/processed/train.csv', parse_dates=['Stay_Date', 'as_of_date'])
    test = pd.read_csv('data/processed/test.csv', parse_dates=['Stay_Date', 'as_of_date'])

    full_data = pd.concat([train, test], ignore_index=True)
    print(f"Full data: {len(full_data):,} samples")
    print(f"Stay date range: {full_data['Stay_Date'].min()} to {full_data['Stay_Date'].max()}")

    return full_data


# ============================================================================
# Create Train/Val/Test Split
# ============================================================================

def create_train_val_test_split(df):
    """
    Split by stay dates:
    - Train: Up to end of 2023
    - Val: 2024 (for hyperparameter tuning)
    - Test: 2025 (final evaluation)
    """

    train_cutoff = pd.Timestamp('2023-12-31')
    val_cutoff = pd.Timestamp('2024-12-31')

    train = df[df['Stay_Date'] <= train_cutoff].copy()
    val = df[(df['Stay_Date'] > train_cutoff) & (df['Stay_Date'] <= val_cutoff)].copy()
    test = df[df['Stay_Date'] > val_cutoff].copy()

    print("\n" + "="*80)
    print("TRAIN/VAL/TEST SPLIT")
    print("="*80)
    print(f"Train: {train['Stay_Date'].min()} to {train['Stay_Date'].max()}")
    print(f"       {len(train):,} samples, {train['Stay_Date'].nunique()} unique stay dates")
    print(f"\nVal:   {val['Stay_Date'].min()} to {val['Stay_Date'].max()}")
    print(f"       {len(val):,} samples, {val['Stay_Date'].nunique()} unique stay dates")
    print(f"\nTest:  {test['Stay_Date'].min()} to {test['Stay_Date'].max()}")
    print(f"       {len(test):,} samples, {test['Stay_Date'].nunique()} unique stay dates")

    return train, val, test


# ============================================================================
# Additive Pickup with Lookback Window
# ============================================================================

def additive_pickup_with_lookback(train_df, test_df, lookback_years=None, verbose=True):
    """
    Additive pickup baseline with configurable lookback window

    Args:
        train_df: Training data (historical stay dates)
        test_df: Test/validation data (future stay dates to forecast)
        lookback_years: How many years of history to use (None = all available)
        verbose: Print details

    Returns:
        test_forecast: Predictions
        metrics: Dict with MAE, RMSE, MAPE
    """

    # Filter training data by lookback window
    if lookback_years is not None:
        cutoff_date = train_df['Stay_Date'].max() - pd.DateOffset(years=lookback_years)
        train_filtered = train_df[train_df['Stay_Date'] >= cutoff_date].copy()
        if verbose:
            print(f"\nUsing {lookback_years} year(s) of history: {train_filtered['Stay_Date'].min()} to {train_filtered['Stay_Date'].max()}")
            print(f"Filtered to {len(train_filtered):,} samples ({train_filtered['Stay_Date'].nunique()} stay dates)")
    else:
        train_filtered = train_df.copy()
        if verbose:
            print(f"\nUsing ALL available history: {train_filtered['Stay_Date'].min()} to {train_filtered['Stay_Date'].max()}")

    # Calculate final occupancy for each stay date
    finals = (train_filtered.sort_values('as_of_date')
                           .groupby(['Property_Code', 'Stay_Date'])['rooms_on_books']
                           .last()
                           .rename('final_occ')
                           .reset_index())

    # Merge and calculate remaining pickup
    train_enriched = train_filtered.merge(finals, on=['Property_Code', 'Stay_Date'], how='left')
    train_enriched['remaining'] = train_enriched['final_occ'] - train_enriched['rooms_on_books']

    # Average remaining pickup by (property, days_to_arrival)
    baseline_pickup = (train_enriched.groupby(['Property_Code', 'days_to_arrival'])['remaining']
                                    .mean()
                                    .rename('avg_remaining')
                                    .reset_index())

    # Apply to test set
    test_forecast = test_df.merge(baseline_pickup, on=['Property_Code', 'days_to_arrival'], how='left')
    test_forecast['avg_remaining'] = test_forecast['avg_remaining'].fillna(0)
    test_forecast['forecast'] = test_forecast['rooms_on_books'] + test_forecast['avg_remaining']
    test_forecast['forecast'] = test_forecast['forecast'].clip(lower=0)

    # Calculate metrics
    mae = mean_absolute_error(test_forecast['rooms_on_books'], test_forecast['forecast'])
    rmse = np.sqrt(mean_squared_error(test_forecast['rooms_on_books'], test_forecast['forecast']))

    # MAPE
    non_zero = test_forecast['rooms_on_books'] > 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((test_forecast.loc[non_zero, 'rooms_on_books'] -
                               test_forecast.loc[non_zero, 'forecast']) /
                              test_forecast.loc[non_zero, 'rooms_on_books'])) * 100
    else:
        mape = np.nan

    if verbose:
        print(f"  MAE:  {mae:.2f} rooms")
        print(f"  RMSE: {rmse:.2f} rooms")
        print(f"  MAPE: {mape:.2f}%")

    return test_forecast, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ============================================================================
# Hyperparameter Tuning on Validation Set
# ============================================================================

def tune_lookback_window(train_df, val_df):
    """
    Try different lookback windows on validation set
    Find best hyperparameter
    """

    print("\n" + "="*80)
    print("TUNING LOOKBACK WINDOW ON VALIDATION SET")
    print("="*80)

    # Try different lookback windows
    lookback_options = [1, 2, 3, 5, None]  # 1yr, 2yr, 3yr, 5yr, all
    results = []

    for lookback in lookback_options:
        label = f"{lookback} year(s)" if lookback else "All available"
        print(f"\n{'─'*80}")
        print(f"Testing lookback: {label}")
        print('─'*80)

        _, metrics = additive_pickup_with_lookback(train_df, val_df, lookback_years=lookback, verbose=True)

        results.append({
            'lookback_years': lookback if lookback else 'all',
            'lookback_label': label,
            **metrics
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    # Best model
    best = results_df.iloc[0]
    print("\n" + "="*80)
    print(f"🏆 BEST LOOKBACK WINDOW: {best['lookback_label']}")
    print(f"   Validation MAE: {best['MAE']:.2f} rooms")
    print("="*80)

    # Save results
    os.makedirs('outputs/tuning', exist_ok=True)
    results_df.to_csv('outputs/tuning/lookback_validation_results.csv', index=False)

    return results_df, best['lookback_years']


# ============================================================================
# Final Evaluation on Test Set
# ============================================================================

def evaluate_on_test(train_df, val_df, test_df, best_lookback):
    """
    Retrain on train+val with best lookback, evaluate on test
    """

    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    # Combine train + val for final training
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\nCombined train+val: {len(train_full):,} samples")
    print(f"Stay dates: {train_full['Stay_Date'].min()} to {train_full['Stay_Date'].max()}")

    # Train with best lookback
    lookback_val = None if best_lookback == 'all' else best_lookback
    label = f"{lookback_val} year(s)" if lookback_val else "All available"

    print(f"\nUsing best lookback window: {label}")
    print('─'*80)

    test_forecast, metrics = additive_pickup_with_lookback(
        train_full, test_df,
        lookback_years=lookback_val,
        verbose=True
    )

    # Save final results
    test_forecast.to_csv('outputs/predictions/baseline_final_test.csv', index=False)

    metrics_df = pd.DataFrame([{
        'model': 'Additive Pickup (Tuned)',
        'lookback_years': best_lookback,
        'split': 'test',
        **metrics
    }])
    metrics_df.to_csv('outputs/metrics/baseline_final_metrics.csv', index=False)

    print("\n" + "="*80)
    print("✅ FINAL TEST RESULTS")
    print("="*80)
    print(f"MAE:  {metrics['MAE']:.2f} rooms")
    print(f"RMSE: {metrics['RMSE']:.2f} rooms")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("\nThis is the baseline to beat with Chronos!")

    return test_forecast, metrics


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("BASELINE HYPERPARAMETER TUNING")
    print("Train/Val/Test Split with Lookback Window Optimization")
    print("="*80)

    # Load data
    full_data = load_processed_data()

    # Create splits
    train_df, val_df, test_df = create_train_val_test_split(full_data)

    # Tune on validation set
    val_results, best_lookback = tune_lookback_window(train_df, val_df)

    # Final evaluation on test set
    test_forecast, test_metrics = evaluate_on_test(train_df, val_df, test_df, best_lookback)

    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Best lookback window: {best_lookback}")
    print(f"Test MAE: {test_metrics['MAE']:.2f} rooms")
    print(f"\nFiles saved:")
    print(f"  - Validation results: outputs/tuning/lookback_validation_results.csv")
    print(f"  - Final test predictions: outputs/predictions/baseline_final_test.csv")
    print(f"  - Final metrics: outputs/metrics/baseline_final_metrics.csv")


if __name__ == '__main__':
    main()
