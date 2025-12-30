# Quick Start Guide - 5 Minutes to First Model

This guide gets you from zero to a working booking curve forecast in 5 minutes.

## Step 1: Setup (1 minute)

```bash
# Navigate to project directory
cd /Users/hetalksinmaths/autogluon-assistant/ayana-booking-forecasting

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install AutoGluon (this may take 2-3 minutes)
pip install autogluon pandas numpy scikit-learn matplotlib
```

## Step 2: Verify Data (30 seconds)

Your data is already at:
```
/Users/hetalksinmaths/Downloads/filtered_hotels.csv
```

Quick check:
```bash
head -2 /Users/hetalksinmaths/Downloads/filtered_hotels.csv
```

You should see columns like: `Stay_Date`, `Property_Code`, `Dmin300`, `Dmin299`, etc.

## Step 3: Explore Data (Optional - 2 minutes)

```bash
python notebooks/01_data_exploration.py
```

This shows you:
- Number of properties and stay dates
- Booking window coverage
- Missing value analysis
- Basic statistics

## Step 4: Run Baseline Model (1 minute)

Test with a simple baseline first:

```bash
# First process the data
python run_autogluon.py
```

Wait for it to process and split the data (this happens first, before training).

Then interrupt it (Ctrl+C) after you see "✓ Saved processed data to ./data/processed/"

Now run baseline:
```bash
python baseline_models.py
```

This gives you a quick baseline MAE to beat.

Expected output:
```
📊 Results:
  MAE:  2.5-4.0 rooms
  RMSE: 3.5-5.5 rooms
  MAPE: 10-15%
```

## Step 5: Run AutoGluon (30-60 minutes)

Now train the full AutoGluon ensemble:

```bash
python run_autogluon.py
```

This will:
1. ✅ Load your Ayana data (1 min)
2. ✅ Transform wide → long format (1 min)
3. ✅ Split train/test (1 min)
4. ⏳ Train models (30-60 min)
5. ✅ Generate predictions (2 min)
6. ✅ Evaluate and save results (1 min)

**Go get coffee while it trains!**

## Step 6: Check Results (1 minute)

After training completes:

```bash
# View evaluation metrics
cat outputs/metrics/evaluation_metrics.csv

# View model leaderboard (see which models performed best)
tail -20 outputs/models/ag_models/leaderboard.csv

# View predictions
head outputs/predictions/autogluon_predictions.csv
```

Expected AutoGluon MAE: **2.0-3.0 rooms** (better than baseline!)

## What You Just Built

You now have:

1. ✅ **Baseline models** - Simple benchmarks (additive pickup, naive, moving average)
2. ✅ **AutoGluon ensemble** - Automatic ML combining:
   - DeepAR (neural network)
   - PatchTST (transformer)
   - ETS (exponential smoothing)
   - AutoARIMA
   - And more...
3. ✅ **Predictions** - Forecasts for your test set
4. ✅ **Evaluation metrics** - MAE, RMSE, MAPE

## Customization (After First Run)

Edit `run_autogluon.py` to customize:

```python
# Change test size
TEST_HORIZON_DAYS = 60  # Default: 90

# Change forecast horizon
PREDICTION_LENGTH = 14  # Default: 30 (days ahead)

# Change training time
TIME_LIMIT = 3600  # Default: 1800 seconds (30 min)

# Change quality preset
presets='high_quality'  # Options: fast_training, medium_quality, high_quality, best_quality
```

## Troubleshooting

### "No module named 'autogluon'"
```bash
# Make sure venv is activated
source venv/bin/activate
pip install autogluon
```

### "Memory Error"
```bash
# Reduce data size by sampling
# Edit run_autogluon.py, line 27, add:
df = pd.read_csv(filepath, nrows=10000)
```

### "Takes too long"
```bash
# Use fast training preset
# Edit run_autogluon.py, line 137, change to:
presets='fast_training'
time_limit=600  # 10 minutes
```

### "File not found"
```bash
# Make sure data path is correct
ls -lh /Users/hetalksinmaths/Downloads/filtered_hotels.csv
```

## Next Steps

### 1. Improve Accuracy (Feature Engineering)

Add these features to `run_autogluon.py`:

```python
# After loading data, add:
df['month'] = df['as_of_date'].dt.month
df['day_of_week_num'] = df['as_of_date'].dt.dayofweek
df['is_weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday'])
df['season'] = df['as_of_date'].dt.month.map({
    12: 'winter', 1: 'winter', 2: 'winter',
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'fall', 10: 'fall', 11: 'fall'
})
```

### 2. Try Different Presets

```bash
# Fast (10 min)
presets='fast_training'

# Medium (30-60 min) - DEFAULT
presets='medium_quality'

# High (2-4 hours)
presets='high_quality'

# Best (8+ hours)
presets='best_quality'
```

### 3. Production Deployment

After you have a good model:

```python
# Save the best model
predictor.save('production_model')

# Later, load and predict
from autogluon.timeseries import TimeSeriesPredictor
predictor = TimeSeriesPredictor.load('production_model')
predictions = predictor.predict(new_data)
```

### 4. Try TFT (Advanced)

If you have a GPU and want state-of-the-art:

```bash
# Install PyTorch Forecasting
pip install torch pytorch-lightning pytorch-forecasting

# Create tft_forecast.py (see main documentation)
python tft_forecast.py
```

## Questions?

Check the full documentation:
- `BOOKING_CURVE_FORECASTING_SETUP.md` - Complete setup guide
- `README.md` - Project overview
- AutoGluon docs: https://auto.gluon.ai/stable/tutorials/timeseries/

Happy forecasting!
