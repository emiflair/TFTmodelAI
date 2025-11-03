"""
Update Training Data - Consolidate multiple CSV files or append new data

This script helps you:
1. Merge multiple training CSV files into one master file
2. Append new historical data to existing training data
3. Remove duplicates and maintain chronological order
4. Verify data quality after updates

Usage:
    # Merge multiple files
    python update_training_data.py --merge file1.csv file2.csv file3.csv
    
    # Append new data to existing
    python update_training_data.py --append new_data.csv
    
    # Just verify current data
    python update_training_data.py --verify
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def load_and_standardize(csv_path: str) -> pd.DataFrame:
    """Load CSV and standardize column names and formats"""
    print(f"  Loading: {csv_path}")
    
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Standardize column names
    column_map = {
        "time (utc)": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "tick_volume",
        "volume (tick)": "tick_volume",
        "tick_volume": "tick_volume",
    }
    df.rename(columns={col: column_map.get(col, col) for col in df.columns}, inplace=True)
    
    # Verify required columns
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
        utc=True
    )
    
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        numeric_cols.append("tick_volume")
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop invalid rows
    df.dropna(subset=["timestamp"] + numeric_cols, inplace=True)
    
    # Sort by time
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"    ✓ Loaded {len(df)} rows")
    print(f"    ✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    return df


def merge_dataframes(dfs: list) -> pd.DataFrame:
    """Merge multiple dataframes, remove duplicates, sort chronologically"""
    print("\nMerging dataframes...")
    
    # Concatenate all
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined)} total rows")
    
    # Remove duplicates (keep first occurrence)
    before = len(combined)
    combined.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    duplicates = before - len(combined)
    if duplicates > 0:
        print(f"  ✓ Removed {duplicates} duplicate timestamps")
    
    # Sort chronologically
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    print(f"  ✓ Final: {len(combined)} unique rows")
    print(f"  ✓ Date range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")
    
    return combined


def verify_data_quality(df: pd.DataFrame) -> dict:
    """Verify data quality and report statistics"""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    stats = {}
    
    # Basic info
    stats['total_rows'] = len(df)
    stats['start_date'] = df['timestamp'].min()
    stats['end_date'] = df['timestamp'].max()
    stats['date_range_days'] = (stats['end_date'] - stats['start_date']).days
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total rows:        {stats['total_rows']:,}")
    print(f"  Start date:        {stats['start_date']}")
    print(f"  End date:          {stats['end_date']}")
    print(f"  Date range:        {stats['date_range_days']} days")
    
    # Check for gaps
    expected_freq = pd.Timedelta(minutes=15)
    time_diffs = df['timestamp'].diff()
    gaps = time_diffs[time_diffs > expected_freq]
    
    stats['gaps'] = len(gaps)
    stats['largest_gap_hours'] = time_diffs.max().total_seconds() / 3600 if len(gaps) > 0 else 0
    
    print(f"\n🔍 Data Continuity:")
    print(f"  Expected frequency: 15 minutes")
    print(f"  Gaps detected:      {stats['gaps']}")
    if stats['gaps'] > 0:
        print(f"  Largest gap:        {stats['largest_gap_hours']:.1f} hours")
    
    # Price statistics
    print(f"\n💰 Price Range:")
    print(f"  Min close:         ${df['close'].min():.2f}")
    print(f"  Max close:         ${df['close'].max():.2f}")
    print(f"  Mean close:        ${df['close'].mean():.2f}")
    
    # Volume check
    if 'tick_volume' in df.columns:
        zero_volume = (df['tick_volume'] == 0).sum()
        stats['zero_volume_pct'] = (zero_volume / len(df)) * 100
        print(f"\n📈 Volume:")
        print(f"  Zero volume bars:  {zero_volume} ({stats['zero_volume_pct']:.2f}%)")
    
    # Duplicates check
    duplicates = df.duplicated(subset=['timestamp']).sum()
    stats['duplicates'] = duplicates
    print(f"\n🔄 Duplicates:")
    print(f"  Duplicate times:   {duplicates}")
    
    # Data quality score
    quality_score = 100
    if stats['gaps'] > 100:
        quality_score -= 20
    if stats['duplicates'] > 0:
        quality_score -= 30
    if 'zero_volume_pct' in stats and stats['zero_volume_pct'] > 10:
        quality_score -= 20
    
    stats['quality_score'] = max(0, quality_score)
    
    print(f"\n{'='*70}")
    print(f"📝 Quality Score: {stats['quality_score']}/100")
    
    if stats['quality_score'] >= 90:
        print("   ✅ EXCELLENT - Data is high quality")
    elif stats['quality_score'] >= 70:
        print("   ⚠️  GOOD - Minor issues detected")
    else:
        print("   ❌ POOR - Significant data quality issues")
    
    print("="*70 + "\n")
    
    return stats


def save_master_file(df: pd.DataFrame, output_path: str, backup: bool = True):
    """Save master training file with backup"""
    output_path = Path(output_path)
    
    # Create backup of existing file
    if backup and output_path.exists():
        backup_path = output_path.parent / f"{output_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\n💾 Creating backup: {backup_path.name}")
        import shutil
        shutil.copy2(output_path, backup_path)
    
    # Format timestamp for output
    df_out = df.copy()
    df_out['timestamp'] = df_out['timestamp'].dt.strftime('%Y.%m.%d %H:%M:%S')
    
    # Rename columns to match original format
    df_out.rename(columns={
        'timestamp': 'Time (UTC)',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    
    # Save
    print(f"💾 Saving to: {output_path}")
    df_out.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df_out)} rows")
    
    # Calculate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ File size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Update and manage training data for TFT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge multiple CSV files
  python update_training_data.py --merge data1.csv data2.csv data3.csv
  
  # Append new data to existing master file
  python update_training_data.py --append new_2025_data.csv
  
  # Just verify current master file quality
  python update_training_data.py --verify
  
  # Merge and specify output file
  python update_training_data.py --merge file1.csv file2.csv -o XAUUSD_MASTER.csv
        """
    )
    
    parser.add_argument('--merge', nargs='+', metavar='CSV',
                       help='Merge multiple CSV files into one')
    parser.add_argument('--append', metavar='CSV',
                       help='Append new data to existing master file')
    parser.add_argument('--verify', action='store_true',
                       help='Verify data quality of master file')
    parser.add_argument('-o', '--output', default='XAUUSD_15M.csv',
                       help='Output file path (default: XAUUSD_15M.csv)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup of existing file')
    
    args = parser.parse_args()
    
    # Determine master file path
    master_file = Path(args.output)
    
    print("\n" + "="*70)
    print("TFT TRAINING DATA UPDATER")
    print("="*70 + "\n")
    
    # VERIFY MODE
    if args.verify or (not args.merge and not args.append):
        if not master_file.exists():
            print(f"❌ Master file not found: {master_file}")
            print(f"   Create it using --merge or --append")
            return 1
        
        print(f"📂 Verifying: {master_file}")
        df = load_and_standardize(str(master_file))
        verify_data_quality(df)
        return 0
    
    # MERGE MODE
    if args.merge:
        print(f"📦 MERGE MODE: Combining {len(args.merge)} files\n")
        
        dfs = []
        for csv_file in args.merge:
            if not Path(csv_file).exists():
                print(f"❌ File not found: {csv_file}")
                return 1
            dfs.append(load_and_standardize(csv_file))
        
        merged = merge_dataframes(dfs)
        verify_data_quality(merged)
        save_master_file(merged, str(master_file), backup=not args.no_backup)
        
        print("\n✅ SUCCESS! Master training file updated.")
        print(f"   You can now run: python src/training/train_tft.py")
        return 0
    
    # APPEND MODE
    if args.append:
        print(f"➕ APPEND MODE: Adding new data to master file\n")
        
        if not master_file.exists():
            print(f"❌ Master file not found: {master_file}")
            print(f"   Use --merge instead for first-time creation")
            return 1
        
        if not Path(args.append).exists():
            print(f"❌ New data file not found: {args.append}")
            return 1
        
        print("Loading existing master file...")
        master_df = load_and_standardize(str(master_file))
        
        print("\nLoading new data...")
        new_df = load_and_standardize(args.append)
        
        # Check for overlap
        overlap_start = max(master_df['timestamp'].min(), new_df['timestamp'].min())
        overlap_end = min(master_df['timestamp'].max(), new_df['timestamp'].max())
        
        if overlap_start <= overlap_end:
            print(f"\n⚠️  Date overlap detected:")
            print(f"   Overlap period: {overlap_start} → {overlap_end}")
            print(f"   Duplicates will be removed (keeping existing data)")
        
        merged = merge_dataframes([master_df, new_df])
        verify_data_quality(merged)
        save_master_file(merged, str(master_file), backup=not args.no_backup)
        
        print("\n✅ SUCCESS! New data appended to master file.")
        print(f"   You can now retrain: python src/training/train_tft.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
