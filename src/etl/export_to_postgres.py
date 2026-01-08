#!/usr/bin/env python3
"""
Export Time Series Analysis Data to PostgreSQL
==============================================
This script exports all CSV data to PostgreSQL database
Includes: raw data, merged data, and correlation statistics
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import os
import time

# ============================================================================
# Database Configuration
# ============================================================================

# PostgreSQL connection parameters - UPDATE THESE!
DB_CONFIG = {
    'host': 'localhost',      # or your PostgreSQL server address
    'port': 5432,             # default PostgreSQL port
    'database': 'time_series_analysis',  # database name
    'user': 'postgres',  # your PostgreSQL username
    'password': '123'  # your PostgreSQL password
}

# Connection string
CONNECTION_STRING = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ============================================================================
# Data Files Configuration
# ============================================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    'stock_data': 'csv_exports/stock_data_wiki_clean.csv',
    'sp500_data': 'csv_exports/sp500.csv',
    'depression_index': 'csv_exports/depression_index.csv',
    'depression_word_count': 'csv_exports/ccnews_depression_daily_count_final.csv',
    'rainfall_data': 'csv_exports/rainfall.csv',
    'daily_merged_data': 'merged_time_series_data.csv',
    'correlation_statistics': 'correlation_statistics_full.csv'
}

# ============================================================================
# Column Mappings (CSV columns -> PostgreSQL columns)
# ============================================================================

COLUMN_MAPPINGS = {
    'stock_data': {
        'Date': 'date',
        'Ticker': 'ticker',
        'Industry': 'industry',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'close_price',
        'Volume': 'volume'
    },
    'sp500_data': {
        'Date': 'date',
        'Open': 'sp500_open',
        'High': 'sp500_high',
        'Low': 'sp500_low',
        'Close': 'sp500_close',
        'Volume': 'sp500_volume',
        'Return': 'sp500_return',
        'Volatility': 'sp500_volatility'
    },
    'depression_index': {
        'Week': 'week_start_date',
        'depression': 'depression_index'
    },
    'depression_word_count': {
        'date': 'date',
        'depression_word_count': 'word_count'
    },
    'rainfall_data': {
        'date': 'date',
        'rainfall': 'rainfall_mm'
    },
    'daily_merged_data': {
        'date': 'date',
        'avg_stock_open': 'avg_open',
        'avg_stock_high': 'avg_high',
        'avg_stock_low': 'avg_low',
        'avg_stock_close': 'avg_close',
        'avg_stock_volume': 'avg_volume',
        'price_range': 'price_range',
        'sp500_close': 'sp500_close',
        'sp500_return': 'sp500_return',
        'sp500_volatility': 'sp500_volatility',
        'depression_index': 'depression_index',
        'depression_word_count': 'depression_word_count',
        'rainfall': 'rainfall_mm'
    },
    'correlation_statistics': {
        'Variable_X': 'variable_x',
        'Variable_Y': 'variable_y',
        'Correlation': 'correlation',
        'P_Value': 'p_value',
        'R_Squared': 'r_squared',
        'CI_Lower': 'ci_lower',
        'CI_Upper': 'ci_upper',
        'N': 'sample_size',
        'Significance': 'significance_level',
        'Effect_Size': 'effect_size',
        'Bonferroni_Significant': 'bonferroni_significant',
        'Importance_Score': 'importance_score'
    }
}

# ============================================================================
# Main Export Function
# ============================================================================

def export_to_postgres():
    """Export all CSV files to PostgreSQL database"""
    
    print("="*70)
    print("EXPORTING TIME SERIES ANALYSIS DATA TO POSTGRESQL")
    print("="*70)
    
    # Create database engine
    try:
        print(f"\nConnecting to database: {DB_CONFIG['database']}")
        print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        engine = create_engine(CONNECTION_STRING)
        
        # Test connection
        with engine.connect() as conn:
            print("✓ Database connection successful!")
            
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        print("\nPlease update DB_CONFIG in this script with your PostgreSQL credentials.")
        return
    
    # Export each table
    total_rows = 0
    successful_tables = 0
    
    for table_name, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        print(f"\n{'-'*70}")
        print(f"Processing: {table_name}")
        print(f"File: {filename}")
        
        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            continue
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            print(f"  Loaded {len(df):,} rows from CSV")
            
            # Rename columns if mapping exists
            if table_name in COLUMN_MAPPINGS:
                df = df.rename(columns=COLUMN_MAPPINGS[table_name])
                print(f"  Renamed columns: {len(COLUMN_MAPPINGS[table_name])} mappings")
            
            # Convert date columns
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for date_col in date_columns:
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"  Converted {date_col} to datetime")
            
            # Special handling for specific tables
            if table_name == 'daily_merged_data':
                # Add num_stocks_traded if not present
                if 'num_stocks_traded' not in df.columns:
                    df['num_stocks_traded'] = None
            
            # Export to PostgreSQL with small batch size to avoid parameter limit issues
            try:
                batch_size = 100  # Small batch size to avoid SQL parameter limit
                total_batches = (len(df) + batch_size - 1) // batch_size
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    batch.to_sql(
                        name=table_name,
                        con=engine,
                        if_exists='append' if i > 0 else 'replace',  # Replace first batch, append rest
                        index=False,
                        method='multi'
                    )
                    batch_num = i//batch_size + 1
                    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} rows)")
                    
                    # Small delay to prevent overwhelming the database
                    time.sleep(0.05)
                
                print(f"✓ Exported {len(df):,} rows to {table_name} in {total_batches} batches")
                total_rows += len(df)
                successful_tables += 1
                
            except Exception as e:
                print(f"✗ Error during batch export: {e}")
                # Try single row insert as fallback (very slow but works)
                print(f"  Attempting single-row fallback...")
                try:
                    for idx, row in df.iterrows():
                        row_df = pd.DataFrame([row])
                        row_df.to_sql(
                            name=table_name,
                            con=engine,
                            if_exists='append' if idx > 0 else 'replace',
                            index=False,
                            method='multi'
                        )
                        if idx % 100 == 0:
                            print(f"    Processed {idx+1}/{len(df)} rows...")
                    print(f"✓ Exported {len(df):,} rows to {table_name} (single-row fallback)")
                    total_rows += len(df)
                    successful_tables += 1
                except Exception as e2:
                    print(f"✗ Fallback also failed: {e2}")
                    continue
            
        except Exception as e:
            print(f"✗ Error exporting {table_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPORT SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successfully exported {successful_tables}/{len(FILES)} tables")
    print(f"✓ Total rows exported: {total_rows:,}")
    print(f"\nDatabase: {DB_CONFIG['database']}")
    print(f"Connection: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    
    # Run verification query
    try:
        with engine.connect() as conn:
            print(f"\n{'='*70}")
            print("VERIFICATION - Row Counts in Database")
            print(f"{'='*70}")
            
            for table_name in FILES.keys():
                try:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = result.fetchone()[0]
                    print(f"{table_name:30s}: {count:,} rows")
                except:
                    print(f"{table_name:30s}: Error querying")
            
            print(f"\n{'='*70}")
            print("TOP 3 CORRELATIONS (Verification)")
            print(f"{'='*70}")
            
            query = """
            SELECT 
                variable_x,
                variable_y,
                correlation,
                p_value,
                bonferroni_significant,
                importance_score
            FROM correlation_statistics
            ORDER BY importance_score DESC, ABS(correlation) DESC
            LIMIT 3
            """
            
            result = pd.read_sql(query, engine)
            print(result.to_string(index=False))
            
    except Exception as e:
        print(f"✗ Error running verification: {e}")
    
    print(f"\n{'='*70}")
    print("✓ EXPORT COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Run queries using psql or any PostgreSQL client")
    print("2. Use the views created in the schema:")
    print("   - v_important_correlations")
    print("   - v_significant_correlations")
    print("   - v_daily_analysis")
    print("   - v_depression_metrics")
    print("3. Explore data with sample queries from create_timeseries_postgres_schema.sql")


# ============================================================================
# Alternative: Export Using psycopg2 (if SQLAlchemy not available)
# ============================================================================

def export_using_psycopg2():
    """Alternative export method using psycopg2 directly"""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        print("psycopg2 not installed. Install with: pip install psycopg2-binary")
        return
    
    print("Using psycopg2 for export...")
    
    try:
        # Connect
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        print("✓ Connected to database")
        
        # Export correlation_statistics as example
        filepath = os.path.join(DATA_DIR, 'correlation_statistics_full.csv')
        df = pd.read_csv(filepath)
        
        # Prepare data
        columns = COLUMN_MAPPINGS['correlation_statistics']
        df = df.rename(columns=columns)
        
        # Build INSERT query
        col_names = ', '.join(columns.values())
        insert_query = f"""
        INSERT INTO correlation_statistics ({col_names})
        VALUES %s
        """
        
        # Convert DataFrame to list of tuples
        values = [tuple(row) for row in df.values]
        
        # Execute batch insert
        execute_values(cursor, insert_query, values)
        conn.commit()
        
        print(f"✓ Exported {len(df)} rows to correlation_statistics")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# Run Export
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TIME SERIES ANALYSIS - POSTGRESQL EXPORT TOOL")
    print("="*70)
    print("\n⚠️  IMPORTANT: Update DB_CONFIG with your PostgreSQL credentials!")
    print("\nCurrent configuration:")
    print(f"  Host: {DB_CONFIG['host']}")
    print(f"  Port: {DB_CONFIG['port']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  User: {DB_CONFIG['user']}")
    
    # Auto-proceed to avoid manual input
    print("\nProceeding with export automatically...")
    export_to_postgres()
