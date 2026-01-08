#!/usr/bin/env python3
"""
Sample script to load your CSV data into MongoDB for testing the query interface
"""

import pandas as pd
import os
from pymongo import MongoClient

def load_csv_to_mongo():
    """Load the merged time series data into MongoDB"""
    try:
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['stock_analysis_db']
        
        # Get the parent directory where data files are located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Load the merged time series data
        csv_file = os.path.join(parent_dir, 'merged_time_series_data.csv')
        print(f"Loading data from: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from CSV")
        
        # Convert DataFrame to dictionary records
        records = df.to_dict('records')
        
        # Clean up the records (handle NaN values)
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        # Insert into MongoDB collection
        collection = db['time_series_data']
        
        # Clear existing data
        collection.delete_many({})
        print("Cleared existing data")
        
        # Insert new data
        result = collection.insert_many(records)
        print(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
        
        # Create some additional sample collections for testing
        
        # Sample companies collection
        companies_data = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "market_cap": 3000000000000},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "market_cap": 2800000000000},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "market_cap": 1700000000000},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "market_cap": 1500000000000},
            {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary", "market_cap": 800000000000},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services", "market_cap": 500000000000},
            {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare", "market_cap": 450000000000},
            {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services", "market_cap": 400000000000},
            {"symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer Staples", "market_cap": 380000000000},
            {"symbol": "HD", "name": "The Home Depot", "sector": "Consumer Discretionary", "market_cap": 360000000000}
        ]
        
        companies_collection = db['companies']
        companies_collection.delete_many({})
        companies_collection.insert_many(companies_data)
        print(f"Inserted {len(companies_data)} companies")
        
        # Sample economic indicators collection
        economic_indicators = []
        for i, record in enumerate(records[:50]):  # Use first 50 records
            if record.get('date'):
                economic_indicators.append({
                    "date": record['date'],
                    "unemployment_rate": 3.7 + (i * 0.1) % 2,
                    "inflation_rate": 2.1 + (i * 0.05) % 1.5,
                    "fed_rate": 5.25 + (i * 0.02) % 0.75,
                    "gdp_growth": 2.8 + (i * 0.03) % 1.2
                })
        
        indicators_collection = db['economic_indicators']
        indicators_collection.delete_many({})
        indicators_collection.insert_many(economic_indicators)
        print(f"Inserted {len(economic_indicators)} economic indicators")
        
        print("\nCollections created:")
        print("- time_series_data: Main dataset with depression index, stock prices, etc.")
        print("- companies: Company information with sectors and market caps")
        print("- economic_indicators: Economic data like unemployment, inflation, etc.")
        
        print("\nSample queries you can try:")
        print('1. Find all records: {}')
        print('2. High depression periods: {"depression_index": {"$gt": 70}}')
        print('3. Recent dates: {"date": {"$gte": "2023-01-01"}}')
        print('4. Technology companies: {"sector": "Technology"}')
        print('5. Large market cap: {"market_cap": {"$gt": 1000000000000}}')
        
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    load_csv_to_mongo()