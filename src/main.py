#!/usr/bin/env python3
"""
Snowflake to Databricks Iceberg Migration Mini Project

This script demonstrates:
1. Simulated Snowflake data extraction
2. Migration to Databricks (simulated as Delta/Iceberg tables)
3. Iceberg table management concepts
4. Governance through data quality checks
5. Incremental processing
6. Validation
7. Analytics/AI usage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directories
DATA_DIR = "data"
ICEBERG_DIR = os.path.join(DATA_DIR, "iceberg_tables")
GOVERNANCE_DIR = os.path.join(DATA_DIR, "governance")

for dir_path in [DATA_DIR, ICEBERG_DIR, GOVERNANCE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class SnowflakeSimulator:
    """Simulates Snowflake data extraction"""

    def __init__(self):
        self.base_date = datetime(2023, 1, 1)

    def generate_customer_data(self, num_records=1000):
        """Generate simulated customer data from Snowflake"""
        np.random.seed(42)

        data = {
            'customer_id': range(1, num_records + 1),
            'name': [f'Customer_{i}' for i in range(1, num_records + 1)],
            'email': [f'customer_{i}@example.com' for i in range(1, num_records + 1)],
            'age': np.random.randint(18, 80, num_records),
            'income': np.random.normal(50000, 15000, num_records).astype(int),
            'signup_date': [self.base_date + timedelta(days=np.random.randint(0, 365))
                           for _ in range(num_records)],
            'last_login': [self.base_date + timedelta(days=np.random.randint(0, 365))
                          for _ in range(num_records)],
            'is_active': np.random.choice([True, False], num_records, p=[0.8, 0.2])
        }

        return pd.DataFrame(data)

    def generate_transaction_data(self, num_records=5000):
        """Generate simulated transaction data"""
        np.random.seed(123)

        customer_ids = range(1, 1001)  # Reference customer data
        products = ['Widget_A', 'Widget_B', 'Widget_C', 'Service_X', 'Service_Y']

        data = {
            'transaction_id': range(1, num_records + 1),
            'customer_id': np.random.choice(customer_ids, num_records),
            'product': np.random.choice(products, num_records),
            'amount': np.random.exponential(100, num_records).round(2),
            'transaction_date': [self.base_date + timedelta(days=np.random.randint(0, 365))
                                for _ in range(num_records)],
            'status': np.random.choice(['completed', 'pending', 'failed'], num_records,
                                     p=[0.9, 0.08, 0.02])
        }

        return pd.DataFrame(data)

class IcebergTableManager:
    """Simulates Iceberg table management"""

    def __init__(self, base_path):
        self.base_path = base_path
        self.metadata = {}

    def create_table(self, table_name, df, partition_by=None):
        """Create an Iceberg table (simulated with Parquet)"""
        table_path = os.path.join(self.base_path, table_name)
        os.makedirs(table_path, exist_ok=True)

        # Save data as Parquet (simulating Iceberg data files)
        data_file = os.path.join(table_path, "data.parquet")
        df.to_parquet(data_file, index=False)

        # Create metadata (simulating Iceberg metadata)
        metadata = {
            'table_name': table_name,
            'schema': df.dtypes.to_dict(),
            'num_records': len(df),
            'created_at': datetime.now().isoformat(),
            'partition_by': partition_by,
            'data_files': [data_file]
        }

        metadata_file = os.path.join(table_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.metadata[table_name] = metadata
        print(f"Created Iceberg table: {table_name}")

    def append_data(self, table_name, new_df):
        """Append data to existing table (simulating Iceberg append)"""
        if table_name not in self.metadata:
            raise ValueError(f"Table {table_name} does not exist")

        table_path = os.path.join(self.base_path, table_name)
        existing_data_file = os.path.join(table_path, "data.parquet")

        # Read existing data
        existing_df = pd.read_parquet(existing_data_file)

        # Append new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Save combined data
        combined_df.to_parquet(existing_data_file, index=False)

        # Update metadata
        self.metadata[table_name]['num_records'] = len(combined_df)
        self.metadata[table_name]['last_modified'] = datetime.now().isoformat()

        metadata_file = os.path.join(table_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata[table_name], f, indent=2, default=str)

        print(f"Appended {len(new_df)} records to {table_name}")

    def query_table(self, table_name, filters=None):
        """Query table with optional filters"""
        if table_name not in self.metadata:
            raise ValueError(f"Table {table_name} does not exist")

        data_file = os.path.join(self.base_path, table_name, "data.parquet")
        df = pd.read_parquet(data_file)

        if filters:
            for col, condition in filters.items():
                if isinstance(condition, dict):
                    if 'min' in condition:
                        df = df[df[col] >= condition['min']]
                    if 'max' in condition:
                        df = df[df[col] <= condition['max']]
                else:
                    df = df[df[col] == condition]

        return df

class GovernanceManager:
    """Handles data governance and quality checks"""

    def __init__(self, governance_dir):
        self.governance_dir = governance_dir
        self.quality_checks = []

    def add_quality_check(self, check_name, check_func):
        """Add a data quality check"""
        self.quality_checks.append({
            'name': check_name,
            'function': check_func
        })

    def run_quality_checks(self, df, table_name):
        """Run all quality checks on a dataframe"""
        results = {
            'table_name': table_name,
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }

        for check in self.quality_checks:
            try:
                result = check['function'](df)
                results['checks'].append({
                    'check_name': check['name'],
                    'passed': bool(result['passed']),  # Convert numpy bool to Python bool
                    'details': result.get('details', '')
                })
            except Exception as e:
                results['checks'].append({
                    'check_name': check['name'],
                    'passed': False,
                    'details': f"Error: {str(e)}"
                })

        # Save results
        results_file = os.path.join(self.governance_dir, f"{table_name}_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

class IncrementalProcessor:
    """Handles incremental data processing"""

    def __init__(self, checkpoint_file, base_date):
        self.checkpoint_file = checkpoint_file
        self.base_date = base_date
        self.last_processed_timestamp = self.load_checkpoint()

    def load_checkpoint(self):
        """Load last processed timestamp"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['last_processed'])
        return self.base_date

    def save_checkpoint(self, timestamp):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'last_processed': timestamp.isoformat()
            }, f)

    def get_incremental_data(self, full_df, timestamp_column):
        """Get data newer than last checkpoint"""
        incremental_df = full_df[full_df[timestamp_column] > self.last_processed_timestamp]
        if not incremental_df.empty:
            self.last_processed_timestamp = incremental_df[timestamp_column].max()
            self.save_checkpoint(self.last_processed_timestamp)
        return incremental_df

class ValidationManager:
    """Handles data validation"""

    def __init__(self):
        self.validation_rules = []

    def add_validation_rule(self, rule_name, rule_func):
        """Add a validation rule"""
        self.validation_rules.append({
            'name': rule_name,
            'function': rule_func
        })

    def validate_data(self, df, source_name):
        """Validate data against all rules"""
        results = {
            'source': source_name,
            'timestamp': datetime.now().isoformat(),
            'validations': []
        }

        for rule in self.validation_rules:
            try:
                result = rule['function'](df)
                results['validations'].append({
                    'rule_name': rule['name'],
                    'passed': bool(result['passed']),  # Convert numpy bool to Python bool
                    'message': result.get('message', '')
                })
            except Exception as e:
                results['validations'].append({
                    'rule_name': rule['name'],
                    'passed': False,
                    'message': f"Error: {str(e)}"
                })

        return results

class AnalyticsAI:
    """Handles analytics and AI usage"""

    def __init__(self):
        self.models = {}

    def customer_segmentation(self, customer_df):
        """Perform customer segmentation using ML"""
        # Prepare features
        features = ['age', 'income']
        X = customer_df[features].copy()

        # Simple segmentation based on income and age
        customer_df['segment'] = pd.cut(customer_df['income'],
                                       bins=[0, 30000, 60000, 100000, float('inf')],
                                       labels=['Low', 'Medium', 'High', 'Premium'])

        return customer_df

    def churn_prediction(self, customer_df, transaction_df):
        """Predict customer churn"""
        # Create transaction summary first
        transaction_summary = transaction_df.groupby('customer_id').agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).rename(columns={'amount': 'total_spent', 'transaction_id': 'transaction_count'})

        # Merge with customers
        merged_df = customer_df.merge(
            transaction_summary,
            on='customer_id', how='left'
        ).fillna({'total_spent': 0, 'transaction_count': 0})

        # Create target: customers with no transactions in last 30 days are at risk
        # Use a more realistic churn definition
        last_transaction = transaction_df.groupby('customer_id')['transaction_date'].max()
        days_since_last_transaction = (datetime.now() - last_transaction).dt.days
        churn_risk = days_since_last_transaction > 30
        churn_risk = churn_risk.reindex(customer_df['customer_id']).fillna(False)

        # Set customer_id as index for proper alignment
        merged_df = merged_df.set_index('customer_id')
        merged_df['churn_risk'] = churn_risk.astype(int)
        merged_df = merged_df.reset_index()

        # Ensure we have both classes
        if merged_df['churn_risk'].nunique() < 2:
            # If all are the same, create some artificial variety for demo
            np.random.seed(42)
            churn_indices = np.random.choice(len(merged_df), size=int(len(merged_df) * 0.2), replace=False)
            merged_df.loc[churn_indices, 'churn_risk'] = 1

        # Simple model features
        features = ['age', 'income', 'total_spent', 'transaction_count']
        X = merged_df[features]
        y = merged_df['churn_risk']

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get probabilities safely
        proba = model.predict_proba(X)
        churn_probabilities = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        return {
            'model': model,
            'accuracy': report['accuracy'],
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'predictions': churn_probabilities
        }

    def generate_insights(self, customer_df, transaction_df):
        """Generate business insights"""
        insights = {}

        # Revenue insights
        total_revenue = transaction_df['amount'].sum()
        insights['total_revenue'] = total_revenue

        # Customer insights
        active_customers = customer_df[customer_df['is_active']].shape[0]
        insights['active_customers'] = active_customers

        # Product performance
        product_revenue = transaction_df.groupby('product')['amount'].sum().sort_values(ascending=False)
        insights['top_products'] = product_revenue.head(3).to_dict()

        return insights

def main():
    print("🚀 Starting Snowflake to Databricks Iceberg Migration Demo")
    print("=" * 60)

    # Initialize components
    snowflake = SnowflakeSimulator()
    iceberg = IcebergTableManager(ICEBERG_DIR)
    governance = GovernanceManager(GOVERNANCE_DIR)
    validation = ValidationManager()
    analytics = AnalyticsAI()

    # Set up governance quality checks
    def check_null_values(df):
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        return {
            'passed': total_nulls == 0,
            'details': f"Found {total_nulls} null values across {len(null_counts[null_counts > 0])} columns"
        }

    def check_data_types(df):
        # Check if numeric columns have valid ranges
        issues = []
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)]
            if not invalid_ages.empty:
                issues.append(f"Invalid ages: {len(invalid_ages)} records")
        if 'income' in df.columns:
            invalid_income = df[df['income'] < 0]
            if not invalid_income.empty:
                issues.append(f"Negative income: {len(invalid_income)} records")

        return {
            'passed': len(issues) == 0,
            'details': '; '.join(issues) if issues else 'All data types valid'
        }

    governance.add_quality_check('null_check', check_null_values)
    governance.add_quality_check('data_type_check', check_data_types)

    # Set up validation rules
    def validate_customer_ids(df):
        if 'customer_id' not in df.columns:
            return {'passed': False, 'message': 'Missing customer_id column'}
        unique_ids = df['customer_id'].nunique()
        total_records = len(df)
        return {
            'passed': unique_ids == total_records,
            'message': f"Unique customer IDs: {unique_ids}/{total_records}"
        }

    validation.add_validation_rule('customer_id_uniqueness', validate_customer_ids)

    print("\n1. 📊 Extracting data from Snowflake (simulated)")
    customer_df = snowflake.generate_customer_data(1000)
    transaction_df = snowflake.generate_transaction_data(5000)

    print(f"   - Customers: {len(customer_df)} records")
    print(f"   - Transactions: {len(transaction_df)} records")

    print("\n2. ✅ Validating extracted data")
    customer_validation = validation.validate_data(customer_df, 'customers')
    transaction_validation = validation.validate_data(transaction_df, 'transactions')

    print(f"   - Customer validation: {'✅ Passed' if all(v['passed'] for v in customer_validation['validations']) else '❌ Failed'}")
    print(f"   - Transaction validation: {'✅ Passed' if all(v['passed'] for v in transaction_validation['validations']) else '❌ Failed'}")

    print("\n3. 🏗️ Creating Iceberg tables in Databricks (simulated)")
    iceberg.create_table('customers', customer_df, partition_by='signup_date')
    iceberg.create_table('transactions', transaction_df, partition_by='transaction_date')

    print("\n4. 🔍 Running governance quality checks")
    customer_quality = governance.run_quality_checks(customer_df, 'customers')
    transaction_quality = governance.run_quality_checks(transaction_df, 'transactions')

    print(f"   - Customer quality: {'✅ Passed' if all(c['passed'] for c in customer_quality['checks']) else '❌ Issues found'}")
    print(f"   - Transaction quality: {'✅ Passed' if all(c['passed'] for c in transaction_quality['checks']) else '❌ Issues found'}")

    print("\n5. 🔄 Incremental processing simulation")
    processor = IncrementalProcessor(os.path.join(DATA_DIR, 'checkpoint.json'), snowflake.base_date)

    # Simulate new data
    new_customers = snowflake.generate_customer_data(50)
    new_transactions = snowflake.generate_transaction_data(200)

    # Process incremental data
    incremental_customers = processor.get_incremental_data(new_customers, 'signup_date')
    incremental_transactions = processor.get_incremental_data(new_transactions, 'transaction_date')

    if not incremental_customers.empty:
        iceberg.append_data('customers', incremental_customers)
        print(f"   - Added {len(incremental_customers)} new customers")

    if not incremental_transactions.empty:
        iceberg.append_data('transactions', incremental_transactions)
        print(f"   - Added {len(incremental_transactions)} new transactions")

    print("\n6. 📈 Analytics and AI Insights")
    analytics_results = analytics.generate_insights(customer_df, transaction_df)
    print(f"   - Total revenue: ${analytics_results['total_revenue']:,.2f}")
    print(f"   - Active customers: {analytics_results['active_customers']}")
    print(f"   - Top products: {analytics_results['top_products']}")

    # Customer segmentation
    segmented_customers = analytics.customer_segmentation(customer_df)
    segment_counts = segmented_customers['segment'].value_counts()
    print(f"   - Customer segments: {segment_counts.to_dict()}")

    # Churn prediction
    churn_results = analytics.churn_prediction(customer_df, transaction_df)
    print(".3f")
    print(f"   - Top churn predictors: {dict(sorted(churn_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3])}")

    print("\n7. 📊 Generating visualizations")
    # Create some basic plots
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    customer_df['age'].hist(bins=20)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.subplot(1, 3, 2)
    transaction_df['amount'].hist(bins=20)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount ($)')
    plt.ylabel('Count')

    plt.subplot(1, 3, 3)
    segment_counts.plot(kind='bar')
    plt.title('Customer Segments')
    plt.xlabel('Segment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'analytics_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("   - Saved analytics plots to data/analytics_plots.png")

    print("\n8. 🎯 Iceberg table queries demonstration")
    # Query examples
    active_customers = iceberg.query_table('customers', {'is_active': True})
    high_value_transactions = iceberg.query_table('transactions',
                                                 {'amount': {'min': 500}})

    print(f"   - Active customers: {len(active_customers)}")
    print(f"   - High-value transactions (>$500): {len(high_value_transactions)}")

    print("\n✅ Migration and processing complete!")
    print("=" * 60)
    print("\nKey concepts demonstrated:")
    print("• Snowflake data extraction (simulated)")
    print("• Iceberg table management (Parquet-based simulation)")
    print("• Data governance and quality checks")
    print("• Incremental processing with checkpoints")
    print("• Data validation")
    print("• Analytics and AI insights")
    print("• Customer segmentation and churn prediction")

if __name__ == "__main__":
    main()