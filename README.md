# sf-to-databricks-iceberg-mini

A minimal demonstration project for Snowflake to Databricks migration with Iceberg table management, showcasing key concepts in data engineering and analytics.

## Overview

This project simulates a complete data pipeline that covers:

1. **Snowflake → Databricks Migration** (simulated)
   - Extract data from simulated Snowflake tables
   - Migrate to Databricks environment (simulated as Iceberg tables)

2. **Iceberg Table Management**
   - Create Iceberg tables with metadata
   - Append data incrementally
   - Query tables with filters

3. **Data Governance**
   - Quality checks (null values, data types)
   - Validation rules
   - Audit trails

4. **Incremental Processing**
   - Checkpoint-based processing
   - Handle new data efficiently

5. **Data Validation**
   - Schema validation
   - Business rule validation

6. **Analytics & AI Usage**
   - Customer segmentation
   - Churn prediction using ML
   - Business insights generation
   - Data visualizations

## Project Structure

```
├── src/
│   └── main.py              # Main demonstration script
├── data/
│   ├── iceberg_tables/      # Simulated Iceberg tables (Parquet format)
│   └── governance/          # Quality check results
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sf-to-databricks-iceberg-mini
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

Execute the main script to see the complete pipeline in action:

```bash
python src/main.py
```

The script will:
- Generate simulated customer and transaction data
- Create Iceberg tables
- Run governance checks
- Perform incremental processing
- Generate analytics and AI insights
- Create visualizations

## Key Components

### SnowflakeSimulator
Simulates data extraction from Snowflake with realistic customer and transaction data.

### IcebergTableManager
Manages Iceberg tables using Parquet files with metadata tracking, demonstrating:
- Table creation with partitioning
- Data append operations
- Query capabilities with filtering

### GovernanceManager
Implements data quality checks and maintains audit trails.

### IncrementalProcessor
Handles incremental data processing using checkpoint files.

### ValidationManager
Validates data against business rules and schema requirements.

### AnalyticsAI
Provides:
- Customer segmentation
- Churn prediction using Random Forest
- Business insights
- Data visualizations

## Output

The script generates:
- Iceberg tables in `data/iceberg_tables/`
- Governance reports in `data/governance/`
- Analytics plots in `data/analytics_plots.png`
- Console output showing the complete pipeline execution

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning for churn prediction
- **matplotlib/seaborn** - Data visualization
- **pyarrow** - Parquet file handling (simulating Iceberg)

## Concepts Demonstrated

This mini-project covers essential data engineering concepts:

- **Data Migration**: ETL processes from source to target systems
- **Lakehouse Architecture**: Combining data lake and warehouse benefits
- **Data Quality**: Ensuring data reliability and consistency
- **Incremental Processing**: Efficient handling of new data
- **Analytics**: Deriving insights from data
- **AI/ML Integration**: Using machine learning for business predictions

## Future Enhancements

For a production system, consider:
- Real Snowflake/Databricks connections
- Apache Iceberg with Spark
- Advanced governance with Apache Atlas
- Real-time streaming
- More sophisticated ML models
- Cloud deployment (AWS/GCP/Azure) 
