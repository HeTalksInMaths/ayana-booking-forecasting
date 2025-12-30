# Ayana Resort Booking Curve Forecasting

Automated machine learning forecasting system for hotel booking curves using AutoGluon.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run AutoGluon Model

```bash
python run_autogluon.py
```

This will:
- Load and transform your Ayana booking curve data
- Split into train/test sets
- Train an ensemble of time series models (AutoGluon)
- Generate predictions
- Save results to `outputs/`

Expected runtime: 30-60 minutes

### 3. Run Baseline Models (Optional)

```bash
python baseline_models.py
```

This compares simple baseline models:
- Additive pickup forecast
- Naive (last value)
- Moving average

Expected runtime: 1-2 minutes

## Project Structure

```
ayana-booking-forecasting/
├── data/
│   ├── raw/              # Original CSV data
│   └── processed/        # Transformed train/test data
├── models/               # Model implementations
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/
│   ├── models/           # Saved models
│   ├── predictions/      # Forecast CSV files
│   └── metrics/          # Evaluation metrics
├── requirements.txt      # Python dependencies
├── run_autogluon.py     # Main AutoGluon script
├── baseline_models.py   # Baseline model implementations
└── README.md            # This file
```

## Data Format

**Input**: Wide format booking curves
- File: `/Users/hetalksinmaths/Downloads/filtered_hotels.csv`
- Columns: `Stay_Date`, `Property_Code`, `Dmin300`...`Dmin001`, `Dplus000`...`Dplus005`

**Processed**: Long format time series
- Columns: `Property_Code`, `Stay_Date`, `as_of_date`, `days_to_arrival`, `rooms_on_books`

## Configuration

Edit these parameters in `run_autogluon.py`:

```python
DATA_PATH = '/Users/hetalksinmaths/Downloads/filtered_hotels.csv'
TEST_HORIZON_DAYS = 90      # Days to hold out for testing
PREDICTION_LENGTH = 30      # Forecast horizon
TIME_LIMIT = 1800           # Training time limit (seconds)
```

## Results

After running, check:
- `outputs/predictions/autogluon_predictions.csv` - Model forecasts
- `outputs/metrics/evaluation_metrics.csv` - MAE, RMSE, MAPE
- `outputs/metrics/baseline_comparison.csv` - Baseline model results

## Troubleshooting

**Memory Error**: Reduce data size or sample fewer rows
**Slow Training**: Reduce `TIME_LIMIT` or use `presets='fast_training'`
**Import Errors**: Make sure you activated the virtual environment

## Next Steps

1. **Feature Engineering**: Add holidays, special events, seasonality
2. **Hyperparameter Tuning**: Try `high_quality` or `best_quality` presets
3. **TFT Model**: Implement Temporal Fusion Transformer for comparison
4. **Production**: Deploy model as API endpoint

## References

- [AutoGluon TimeSeries Documentation](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [Full Setup Guide](../BOOKING_CURVE_FORECASTING_SETUP.md)

## License

MIT
