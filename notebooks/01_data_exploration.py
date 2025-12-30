"""
Data Exploration for Ayana Booking Curves

This notebook explores the Ayana Resort booking curve data
Run this as a script or convert to Jupyter notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("📊 Ayana Resort Booking Curve Data Exploration")
print("=" * 80)

# ============================================================================
# 1. Load Data
# ============================================================================

print("\n1️⃣  Loading Data")
print("-" * 80)

DATA_PATH = '/Users/hetalksinmaths/Downloads/filtered_hotels.csv'
df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"\nFirst few columns: {df.columns.tolist()[:10]}")

# ============================================================================
# 2. Basic Statistics
# ============================================================================

print("\n2️⃣  Basic Statistics")
print("-" * 80)

print(f"Date range: {df['Stay_Date'].min()} to {df['Stay_Date'].max()}")
print(f"\nProperties:")
for prop in df['Property_Code'].unique():
    count = (df['Property_Code'] == prop).sum()
    print(f"  {prop}: {count:,} stay dates")

print(f"\nDay of Week distribution:")
print(df['Day_of_Week'].value_counts().sort_index())

# ============================================================================
# 3. Booking Curve Analysis
# ============================================================================

print("\n3️⃣  Booking Curve Analysis")
print("-" * 80)

# Get all D columns
d_cols = [col for col in df.columns if col.startswith('Dmin') or col.startswith('Dplus')]
print(f"Number of booking curve snapshots: {len(d_cols)}")
print(f"Booking window: D-{len([c for c in d_cols if c.startswith('Dmin')])} to D+{len([c for c in d_cols if c.startswith('Dplus')])-1}")

# Sample booking curve
print("\nSample booking curve (first stay date):")
sample_row = df.iloc[0]
print(f"Stay Date: {sample_row['Stay_Date']}")
print(f"Property: {sample_row['Property_Code']}")
print(f"Day: {sample_row['Day_of_Week']}")

# Show first 10 and last 5 snapshots
print("\nFirst 10 snapshots:")
for col in d_cols[:10]:
    val = sample_row[col]
    if pd.notna(val):
        print(f"  {col}: {val:.0f} rooms")

print("\nLast 5 snapshots:")
for col in d_cols[-5:]:
    val = sample_row[col]
    if pd.notna(val):
        print(f"  {col}: {val:.0f} rooms")

# ============================================================================
# 4. Missing Values
# ============================================================================

print("\n4️⃣  Missing Values Analysis")
print("-" * 80)

missing_counts = df[d_cols].isnull().sum()
total_cells = len(df) * len(d_cols)
total_missing = missing_counts.sum()
missing_pct = (total_missing / total_cells) * 100

print(f"Total booking curve cells: {total_cells:,}")
print(f"Missing values: {total_missing:,} ({missing_pct:.2f}%)")

# Which columns have the most missing?
top_missing = missing_counts.sort_values(ascending=False).head(10)
print(f"\nTop 10 columns with missing values:")
for col, count in top_missing.items():
    pct = (count / len(df)) * 100
    print(f"  {col}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 5. Occupancy Statistics
# ============================================================================

print("\n5️⃣  Occupancy Statistics by Property")
print("-" * 80)

# Final occupancy (Dplus000)
if 'Dplus000' in df.columns:
    final_occ = df.groupby('Property_Code')['Dplus000'].describe()
    print(final_occ)

# ============================================================================
# 6. Data Quality Checks
# ============================================================================

print("\n6️⃣  Data Quality Checks")
print("-" * 80)

# Check for negative values
negative_counts = (df[d_cols] < 0).sum().sum()
print(f"Negative values: {negative_counts}")

# Check for non-monotonic booking curves (rooms decreasing over time)
# This would be unusual - cancellations should be rare
non_monotonic = 0
for idx, row in df.head(100).iterrows():  # Check first 100 for speed
    values = row[d_cols].dropna().values
    if len(values) > 1 and any(np.diff(values) < 0):
        non_monotonic += 1

print(f"Non-monotonic curves in sample of 100: {non_monotonic} (cancellations/corrections)")

# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ Data Exploration Complete")
print("=" * 80)

print(f"""
Summary:
- Properties: {df['Property_Code'].nunique()}
- Stay dates: {len(df):,}
- Booking window: {len([c for c in d_cols if c.startswith('Dmin')])} days before arrival
- Missing data: {missing_pct:.2f}%
- Data ready for modeling: {'✓' if missing_pct < 30 else '⚠️  (high missing rate)'}

Next steps:
1. Transform to long format (wide → long)
2. Handle missing values (forward fill, interpolation, or drop)
3. Create train/test split
4. Train AutoGluon model
""")

# Optional: Save summary statistics
summary_stats = {
    'n_properties': df['Property_Code'].nunique(),
    'n_stay_dates': len(df),
    'n_booking_snapshots': len(d_cols),
    'missing_pct': missing_pct,
    'date_range_start': df['Stay_Date'].min(),
    'date_range_end': df['Stay_Date'].max(),
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('../outputs/metrics/data_summary.csv', index=False)
print("\n💾 Summary saved to: ../outputs/metrics/data_summary.csv")
