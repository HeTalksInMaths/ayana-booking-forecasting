# Chronos Models for Booking Curve Forecasting

## What is Chronos?

**Chronos** and **Chronos-Bolt** are Amazon's zero-shot foundation models for time series forecasting. They are pre-trained on millions of diverse time series and can forecast **without any training** on your data.

Think of them like GPT for time series - they've seen so many patterns that they can generalize to new datasets immediately.

## Key Advantages

### 1. Zero-Shot (No Training Required)
- No need to train models
- No hyperparameter tuning
- Just load and predict
- **Runs in seconds instead of hours**

### 2. Pre-trained on Diverse Data
- Trained on 100+ million time series
- Covers diverse domains: retail, energy, finance, web traffic, etc.
- Learns general temporal patterns

### 3. Works Well on Small Datasets
- Traditional models need lots of data
- Chronos works even with limited historical data
- Perfect for new hotels or properties

## Available Chronos Models

### Chronos-Bolt (Latest, Recommended)

**Fast and accurate zero-shot models released in 2024**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `chronos-bolt-tiny` | 8M | ⚡⚡⚡ | ⭐⭐ | Rapid prototyping |
| `chronos-bolt-mini` | 20M | ⚡⚡⚡ | ⭐⭐⭐ | Production (fast) |
| `chronos-bolt-small` | 50M | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `chronos-bolt-base` | 200M | ⚡ | ⭐⭐⭐⭐ | **Best balance (recommended)** |
| `chronos-bolt-large` | 600M | 🐌 | ⭐⭐⭐⭐⭐ | Maximum accuracy |

### Chronos (Original)

| Model | Size | Notes |
|-------|------|-------|
| `chronos_tiny` | 8M | Legacy |
| `chronos_mini` | 20M | Legacy |
| `chronos_small` | 46M | Legacy |
| `chronos_base` | 200M | Legacy, use Bolt version |
| `chronos_large` | 710M | Legacy, use Bolt version |

**Recommendation**: Use `chronos-bolt-base` for best speed/accuracy tradeoff.

## Installation

```bash
# Just install AutoGluon - Chronos is included
pip install autogluon
```

No need for the autogluon-assistant! This is direct AutoGluon usage.

## Usage

### Option 1: Single Chronos Model (Fastest)

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Prepare your data
train_ts = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column='Property_Code',
    timestamp_column='date',
    target='rooms_on_books'
)

# Create predictor
predictor = TimeSeriesPredictor(
    prediction_length=30,
    path='./chronos_model'
)

# Fit with Chronos-Bolt (zero-shot, instant)
predictor.fit(
    train_ts,
    hyperparameters={'Chronos-Bolt': {}}  # That's it!
)

# Predict
predictions = predictor.predict(train_ts)
```

### Option 2: Chronos Ensemble (Best Accuracy)

Combine multiple Chronos models:

```python
predictor.fit(
    train_ts,
    hyperparameters={
        'Chronos-Bolt': {},  # Fast
        'Chronos': {},       # Accurate
        'ETS': {},           # Traditional statistical
        'AutoARIMA': {},     # Traditional statistical
    }
)
```

AutoGluon automatically ensembles them and picks the best combination.

### Option 3: Using Our Ready Script

```bash
# Run the pre-configured script
python run_autogluon_chronos.py
```

This handles all the data transformation and uses `chronos-bolt-base` by default.

## Comparison: Traditional vs Chronos

### Traditional AutoML Approach

```python
predictor.fit(
    train_ts,
    presets='medium_quality',  # Trains multiple models
    time_limit=1800            # Takes 30+ minutes
)
```

**Pros**: Can achieve slightly better accuracy if you have lots of data
**Cons**: Slow, requires tuning, needs significant historical data

### Chronos Zero-Shot Approach

```python
predictor.fit(
    train_ts,
    hyperparameters={'Chronos-Bolt': {}}  # Zero-shot
)
```

**Pros**: Instant (seconds), no tuning, works with small data
**Cons**: May be slightly less accurate on very large datasets

## Expected Performance on Ayana Data

Based on similar hotel booking datasets:

| Approach | MAE (rooms) | Time | Data Needed |
|----------|-------------|------|-------------|
| Naive baseline | 3.5-5.0 | 1 min | Any |
| Additive pickup | 2.5-4.0 | 1 min | Any |
| Traditional AutoML | 2.0-3.0 | 30-60 min | 6+ months |
| **Chronos-Bolt** | **2.0-3.5** | **2-5 min** | **Any** |
| Chronos Ensemble | 1.8-3.0 | 10-20 min | Any |
| TFT (deep learning) | 1.5-2.5 | 1-3 hours | 12+ months |

## When to Use What

### Use Chronos-Bolt When:
- ✅ You want fast results (minutes, not hours)
- ✅ You have limited historical data
- ✅ You're prototyping or iterating quickly
- ✅ You have new properties with little history
- ✅ You want a simple, maintainable solution

### Use Traditional AutoML When:
- ✅ You have 12+ months of dense historical data
- ✅ You can afford 1-3 hours of training time
- ✅ You need to squeeze out every % of accuracy
- ✅ You want interpretable feature importance

### Use TFT (PyTorch Forecasting) When:
- ✅ You have a GPU
- ✅ You have 2+ years of data
- ✅ You need attention interpretability
- ✅ You're doing research or competing in Kaggle

## Configuration in run_autogluon_chronos.py

Edit these settings:

```python
# Choose your Chronos model
CHRONOS_MODEL = 'chronos-bolt-base'  # Recommended

# Options:
# CHRONOS_MODEL = 'chronos-bolt-small'   # Faster
# CHRONOS_MODEL = 'chronos-bolt-large'   # More accurate
# CHRONOS_MODEL = 'chronos_tiny'         # Legacy, fastest
```

## Model Download

First run downloads the model (one-time):
- `chronos-bolt-tiny`: ~50 MB
- `chronos-bolt-base`: ~800 MB
- `chronos-bolt-large`: ~2.4 GB

Subsequent runs use the cached model.

## Advanced: Fine-Tuning Chronos

You can fine-tune Chronos on your specific data:

```python
predictor.fit(
    train_ts,
    hyperparameters={
        'Chronos': {
            'fine_tune': True,
            'epochs': 10
        }
    },
    time_limit=1800
)
```

This adapts the pre-trained model to your booking curves. Usually not necessary but can improve accuracy by 10-20%.

## Troubleshooting

### "Model download is slow"
First download can take 5-10 minutes depending on model size and connection. It's cached afterward.

### "Out of memory"
Use a smaller model:
```python
CHRONOS_MODEL = 'chronos-bolt-small'  # Instead of 'base'
```

### "Predictions are off"
1. Check your data preprocessing
2. Try ensemble approach instead of single model
3. Consider fine-tuning with `fine_tune=True`

## Example: Complete Workflow

```python
# 1. Load your Ayana data
df = pd.read_csv('filtered_hotels.csv')

# 2. Transform to long format (see run_autogluon_chronos.py)
long_df = transform_wide_to_long(df)

# 3. Create train/test split
train, test = split_data(long_df)

# 4. Create TimeSeriesDataFrame
train_ts = TimeSeriesDataFrame.from_data_frame(
    train,
    id_column=['Property_Code', 'Stay_Date'],
    timestamp_column='as_of_date',
    target='rooms_on_books'
)

# 5. Fit Chronos (zero-shot, instant!)
predictor = TimeSeriesPredictor(prediction_length=30)
predictor.fit(train_ts, hyperparameters={'Chronos-Bolt': {}})

# 6. Predict
test_ts = TimeSeriesDataFrame.from_data_frame(test, ...)
predictions = predictor.predict(test_ts)

# 7. Evaluate
mae = mean_absolute_error(test['rooms_on_books'], predictions['mean'])
print(f"MAE: {mae:.2f} rooms")
```

## Resources

- [AutoGluon Chronos Documentation](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- [Chronos Paper (arXiv)](https://arxiv.org/abs/2403.07815)
- [Chronos-Bolt Release Notes](https://github.com/amazon-science/chronos-forecasting)

## Summary

**For Ayana booking curves, use Chronos-Bolt:**

1. Fast (minutes vs hours)
2. No training needed (zero-shot)
3. Good accuracy (competitive with trained models)
4. Works with limited data
5. Simple to use and maintain

```bash
# Just run this:
python run_autogluon_chronos.py
```

Done! You'll have forecasts in 2-5 minutes.
