# Booking Curve Forecasting: Problem Definition

## Understanding Your Data

### Raw Data Format (Wide)

Your `filtered_hotels.csv` looks like this:

```csv
Stay_Date,Property_Code,Day_of_Week,Dmin300,Dmin299,...,Dmin001,Dplus000,Dplus001,...,Dplus005
"May 23, 2025",ARSB,Friday,1,1,1,2,3,...,20,22,22,...,22
```

**What this represents**: A single row is a "booking curve" - the evolution of bookings for one specific stay date at one property.

### Transformed Data Format (Long)

We transform to:

```csv
Property_Code,Stay_Date,as_of_date,days_to_arrival,rooms_on_books
ARSB,2025-05-23,2024-08-06,-300,1
ARSB,2025-05-23,2024-08-07,-299,1
ARSB,2025-05-23,2024-08-08,-298,1
...
ARSB,2025-05-23,2025-05-22,-1,20
ARSB,2025-05-23,2025-05-23,0,22    ← Arrival day (final occupancy)
ARSB,2025-05-23,2025-05-24,1,22
```

Each row is now a **snapshot** of bookings at a specific point in time.

---

## Features (What We Know)

### Primary Features

1. **Property_Code** (categorical)
   - Which hotel: ARSB, RJB, SEGARA, VARSB
   - Different properties have different booking patterns
   - ARSB might book faster than VARSB

2. **Stay_Date** (date)
   - The arrival date we're forecasting for
   - Seasonal patterns: high season vs low season
   - Special dates: holidays, festivals, etc.

3. **as_of_date** (date)
   - When we're making the observation
   - This is the "current date" in the forecast
   - Think of it as "today's date" when looking at bookings

4. **days_to_arrival** (integer, -300 to +5)
   - How far we are from arrival
   - Negative = before arrival (D-60 = 60 days before)
   - 0 = arrival day
   - Positive = after arrival (for late bookings)
   - **This is the time dimension**

5. **rooms_on_books** (integer, target variable)
   - Cumulative rooms booked as of `as_of_date`
   - This is what we're trying to predict
   - Always increases over time (or stays flat)

### Derived Features (Can Add)

6. **Day_of_Week** (categorical)
   - Friday/Saturday arrivals typically book differently than Monday/Tuesday
   - Weekends might fill faster

7. **Month** (categorical/cyclic)
   - January vs July vs December
   - Seasonality patterns

8. **Is_Holiday** (binary)
   - Special dates: Christmas, New Year, Nyepi (Bali holiday), etc.
   - Booking patterns change dramatically around holidays

9. **Historical Pickup** (numerical)
   - Average pickup for this property at this days_to_arrival
   - Example: "ARSB typically has 50 rooms at D-60"

10. **Days_to_Weekend** (numerical)
    - If stay date is Friday, this helps model weekend patterns

---

## Target (What We're Predicting)

### The Forecasting Task

**Given**: Historical booking curve up to time T
**Predict**: Future booking curve from T+1 to T+H

**Concrete Example:**

Today is **2025-01-01** (D-142 for a May 23 stay date).
Currently we have **10 rooms booked**.

**What we want to predict:**
- How many rooms at D-100? (March 2025)
- How many rooms at D-60? (April 2025)
- How many rooms at D-30? (late April 2025)
- How many rooms at D-7? (mid May 2025)
- **How many rooms at D-0?** (May 23, 2025 - **final occupancy**)

This is a **multi-step time series forecasting** problem.

### Two Common Approaches

#### Approach 1: Multi-Step Forecast (What AutoGluon Does)

```
Input:  Past 60 observations of booking curve
Output: Next 30 time steps of bookings

Example:
- Observe: D-300 to D-60 (240 snapshots)
- Predict: D-59 to D-30 (30 snapshots)
```

**This is what `prediction_length=30` means in the code.**

#### Approach 2: Final Occupancy Forecast (Alternative)

```
Input:  Current state (rooms_on_books, days_to_arrival, property, etc.)
Output: Final occupancy at D-0

Example:
- Given: D-60, 50 rooms booked
- Predict: Final occupancy = 75 rooms
- Implied pickup: 25 rooms between D-60 and D-0
```

**For AutoGluon, we're using Approach 1 (multi-step).**

---

## What We're Optimizing (Loss Function)

### Metrics

1. **MAE (Mean Absolute Error)** - PRIMARY METRIC
   ```
   MAE = average(|predicted_rooms - actual_rooms|)
   ```

   **Example**: If we predict 50 rooms but actual is 55:
   - Error = |50 - 55| = 5 rooms
   - MAE averages this across all predictions

   **Interpretation**:
   - MAE = 2.5 rooms means on average we're off by 2.5 rooms
   - Lower is better
   - Industry target: <3 rooms for hotel forecasting

2. **RMSE (Root Mean Squared Error)** - SECONDARY METRIC
   ```
   RMSE = sqrt(average((predicted_rooms - actual_rooms)²))
   ```

   **Why it matters**: Penalizes large errors more heavily
   - Being off by 10 rooms hurts more than being off by 1 room twice
   - Good for detecting when model completely misses peak demand

3. **MAPE (Mean Absolute Percentage Error)** - RELATIVE METRIC
   ```
   MAPE = average(|predicted - actual| / actual) * 100
   ```

   **Example**: Predict 50, actual is 55
   - Error = |50-55|/55 = 9.1%

   **Why it matters**:
   - 5 room error on 10 rooms booked is BAD (50% error)
   - 5 room error on 100 rooms booked is GOOD (5% error)
   - Captures relative accuracy

### What Success Looks Like

For Ayana Resort booking curves:

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| MAE | 4-5 rooms | 2-3 rooms | <2 rooms |
| RMSE | 6-8 rooms | 3-5 rooms | <3 rooms |
| MAPE | 15-20% | 8-12% | <8% |

---

## The Full Forecasting Flow

### Training Phase

```
For each (Property, Stay_Date) pair:
  1. Split booking curve into history and future
     - History: D-300 to D-60 (train on this)
     - Future:  D-59 to D-0 (validate on this)

  2. Train model to predict:
     Given: sequence of rooms_on_books at D-300, D-299, ..., D-60
     Predict: rooms_on_books at D-59, D-58, ..., D-30

  3. Evaluate MAE between predicted and actual
```

### Inference Phase (Production Use)

```
Today: 2025-03-15 (D-69 for May 23 stay)
Current bookings: 45 rooms

Model predicts:
  D-68: 46 rooms
  D-67: 47 rooms
  ...
  D-30: 62 rooms
  ...
  D-0:  75 rooms (final occupancy forecast)

Revenue manager uses this to:
  - Set pricing strategy
  - Decide whether to accept group bookings
  - Predict final revenue for May 23
```

---

## Why This Problem is Challenging

1. **Non-linear patterns**: Bookings don't increase linearly
   - Slow at D-300 to D-120
   - Accelerates D-90 to D-30
   - Last-minute surge D-7 to D-0

2. **Different properties, different patterns**:
   - ARSB (main resort): Books steadily
   - VARSB (villas): More last-minute bookings
   - Each needs different model parameters

3. **Seasonality**:
   - High season (July-August, Dec-Jan): Fast booking pace
   - Low season (Feb-Mar, Oct-Nov): Slower pace
   - Models must learn seasonal patterns

4. **Special events**:
   - Nyepi (Bali Day of Silence): Unusual patterns
   - Christmas/New Year: Booked 6+ months out
   - Weddings: Large group bookings

5. **Sparse data at long lead times**:
   - D-300: Very few bookings, lots of noise
   - D-30: More bookings, clearer signal
   - Model must handle both

---

## How AutoGluon Handles This

### Traditional AutoML Approach

```python
predictor.fit(
    train_ts,
    presets='medium_quality',
    time_limit=1800
)
```

AutoGluon trains:
1. **ETS** (Exponential Smoothing) - captures trends
2. **AutoARIMA** - captures seasonality and autocorrelation
3. **DeepAR** - neural network, learns complex patterns
4. **PatchTST** - transformer, captures long-range dependencies
5. **Ensembles** them with weighted voting

**The model learns**:
- At D-60, ARSB typically has X% of final occupancy
- Weekends book faster than weekdays
- July fill-in is faster than February
- VARSB has more last-minute pickup than RJB

### Chronos Zero-Shot Approach

```python
predictor.fit(
    train_ts,
    hyperparameters={'Chronos-Bolt': {}}
)
```

Chronos was pre-trained on millions of time series and learned:
- General booking curve shapes (S-curves, exponential growth)
- Seasonal patterns across industries
- Property-specific patterns transfer across hotels

**It generalizes** from seeing booking curves from hotels worldwide.

---

## Summary: The Core Problem

**INPUT (Features)**:
- Which property (ARSB, RJB, etc.)
- Which stay date (May 23, 2025)
- Current booking snapshot (D-60, 50 rooms booked)
- Day of week, season, historical patterns

**OUTPUT (Target)**:
- Future booking curve: [D-59: 51 rooms, D-58: 52 rooms, ..., D-0: 75 rooms]

**OBJECTIVE (Loss)**:
- Minimize MAE: Get as close as possible to actual future bookings
- Target: <3 rooms average error

**BUSINESS IMPACT**:
- Accurate forecasts → Better pricing decisions
- Better pricing → 5-10% more revenue
- For 100-room hotel @ $200/night → $1-2M annual revenue impact

---

## Validation of Understanding

Please confirm:

1. **Is the target always `rooms_on_books` in the future?**
   - Yes → We're doing multi-step time series forecasting
   - No → We might be predicting something else (final occupancy, pickup rate, etc.)

2. **What's your prediction horizon?**
   - At D-60, do you want to forecast all the way to D-0? (60 steps)
   - Or just D-59 to D-30? (30 steps)
   - Or just final occupancy at D-0? (1 value)

3. **What's the business use case?**
   - Daily revenue management (need daily forecasts)
   - Weekly pricing decisions (weekly aggregates OK)
   - Final occupancy prediction (just D-0 needed)

Let me know if this matches your understanding or if we need to adjust the problem formulation!
