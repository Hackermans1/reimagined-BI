from dataclasses import dataclass
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('csv_analysis.log')
    ]
)
logger = logging.getLogger("csv_analyzer")

class CSVData:
    """Class to handle CSV file operations and analysis"""
    
    def __init__(self, df=None, file_path=None):
        """Initialize with either a dataframe or a path to a CSV file"""
        self.file_path = file_path
        
        if df is not None:
            self.df = df
            logger.info("Using provided dataframe")
        elif file_path:
            try:
                self.df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded CSV file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading CSV file: {e}")
                raise ValueError(f"Failed to load CSV file: {e}")
        else:
            # Create test dataframe with sample sales data
            self.df = self.create_test_dataframe()
            logger.info("Created test dataframe")
    
    def create_test_dataframe(self):
        """Create a test dataframe with sample sales data"""
        # Sample data
        data = {
            'Date': pd.date_range(start='2023-01-01', periods=100),
            'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], 100),
            'Category': np.random.choice(['Electronics', 'Accessories', 'Peripherals'], 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Sales': np.random.randint(100, 2000, 100),
            'Quantity': np.random.randint(1, 10, 100),
            'Customer_Age': np.random.randint(18, 65, 100),
            'Customer_Type': np.random.choice(['New', 'Returning', 'Loyal'], 100),
            'Discount': np.random.choice([0, 5, 10, 15, 20], 100),
            'Rating': np.random.uniform(1, 5, 100).round(1)
        }
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Add calculated columns
        df['Unit_Price'] = (df['Sales'] / df['Quantity']).round(2)
        df['Month'] = df['Date'].dt.month_name()
        df['Quarter'] = 'Q' + df['Date'].dt.quarter.astype(str)
        
        return df
    
    def column_names(self) -> List[str]:
        """Get the names of all columns in the CSV"""
        return self.df.columns.tolist()
    
    def column_count(self) -> int:
        """Get the number of columns in the CSV"""
        return len(self.df.columns)
    
    def row_count(self) -> int:
        """Get the number of rows in the CSV"""
        return len(self.df)
    
    def get_stats(self, column_name: str) -> Dict[str, Any]:
        """Get statistical properties of a numerical column"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV")
        
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            raise TypeError(f"Column '{column_name}' is not numerical")
        
        stats = {
            "mean": float(self.df[column_name].mean()),
            "median": float(self.df[column_name].median()),
            "min": float(self.df[column_name].min()),
            "max": float(self.df[column_name].max()),
            "std": float(self.df[column_name].std())
        }
        return stats
    
    def get_sample(self, rows: int = 5) -> Dict[str, List[Any]]:
        """Get a sample of the CSV data"""
        sample = self.df.head(rows).to_dict(orient='list')
        return sample
    
    def get_unique_values(self, column_name: str) -> List[Any]:
        """Get unique values in a column"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV")
        
        return self.df[column_name].unique().tolist()
    
    def get_value_counts(self, column_name: str) -> Dict[str, int]:
        """Get count of unique values in a column"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV")
        
        # Convert to dictionary with string keys for JSON serialization
        counts = self.df[column_name].value_counts().to_dict()
        return {str(k): v for k, v in counts.items()}
    
    def correlation(self, column1: str, column2: str) -> float:
        """Calculate correlation between two numerical columns"""
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValueError(f"Column not found in CSV")
        
        if not (pd.api.types.is_numeric_dtype(self.df[column1]) and 
                pd.api.types.is_numeric_dtype(self.df[column2])):
            raise TypeError("Both columns must be numerical")
        
        return float(self.df[column1].corr(self.df[column2]))
    
    def group_by_stats(self, group_column: str, value_column: str, agg_func: str = 'mean') -> Dict[str, float]:
        """Get aggregated statistics grouped by a column"""
        if group_column not in self.df.columns or value_column not in self.df.columns:
            raise ValueError(f"Column not found in CSV")
            
        if not pd.api.types.is_numeric_dtype(self.df[value_column]):
            raise TypeError(f"Column '{value_column}' must be numerical for aggregation")
            
        valid_agg_funcs = ['mean', 'sum', 'min', 'max', 'count', 'median']
        if agg_func not in valid_agg_funcs:
            raise ValueError(f"Invalid aggregation function. Must be one of {valid_agg_funcs}")
            
        # Group and aggregate
        grouped = getattr(self.df.groupby(group_column)[value_column], agg_func)()
        return grouped.to_dict()
        
    def time_series_analysis(self, date_column: str, value_column: str, freq: str = 'M') -> Dict[str, float]:
        """Perform time series analysis on a date column and value column"""
        if date_column not in self.df.columns or value_column not in self.df.columns:
            raise ValueError(f"Column not found in CSV")
            
        if not pd.api.types.is_datetime64_dtype(self.df[date_column]):
            try:
                # Try to convert to datetime
                self.df[date_column] = pd.to_datetime(self.df[date_column])
            except:
                raise TypeError(f"Column '{date_column}' cannot be converted to datetime")
                
        if not pd.api.types.is_numeric_dtype(self.df[value_column]):
            raise TypeError(f"Column '{value_column}' must be numerical")
            
        # Resample time series data
        time_series = self.df.set_index(date_column)[value_column].resample(freq).sum()
        return time_series.to_dict()
        
    def filter_data(self, column: str, value: Any, operator: str = '==') -> 'CSVData':
        """Filter data based on a condition and return a new CSVData object"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")
            
        # Create filter based on operator
        if operator == '==':
            filtered_df = self.df[self.df[column] == value]
        elif operator == '!=':
            filtered_df = self.df[self.df[column] != value]
        elif operator == '>':
            filtered_df = self.df[self.df[column] > value]
        elif operator == '<':
            filtered_df = self.df[self.df[column] < value]
        elif operator == '>=':
            filtered_df = self.df[self.df[column] >= value]
        elif operator == '<=':
            filtered_df = self.df[self.df[column] <= value]
        elif operator == 'contains':
            filtered_df = self.df[self.df[column].astype(str).str.contains(str(value))]
        else:
            raise ValueError(f"Invalid operator: {operator}")
            
        # Create new CSVData object with filtered data
        new_csv_data = CSVData(df=filtered_df)
        return new_csv_data


@dataclass
class CSVAnalyzerDependencies:
    csv_data: CSVData


class CSVAnalysisResult(BaseModel):
    answer: str = Field(description='Answer to the user question about the CSV')


# Set up the OpenAI model with Ollama as the provider
model = OpenAIModel(
    model_name='llama3.2:3b',  # Using local Llama model via Ollama
    provider=OpenAIProvider(
        base_url='http://localhost:11434/v1'
    ),
)

csv_analyzer_agent = Agent(
    model=model,
    deps_type=CSVAnalyzerDependencies,
    result_type=CSVAnalysisResult,
    retries=3,
    system_prompt=(
        'You are a CSV analysis assistant. Help users understand their CSV data '
        'by answering questions about its contents, structure, and statistical properties. '
        'Be precise and include numerical details when relevant. '
        'When appropriate, suggest follow-up questions that might help the user gain more insights.'
    ),
)


@csv_analyzer_agent.system_prompt
async def add_csv_metadata(ctx: RunContext[CSVAnalyzerDependencies]) -> str:
    """Add metadata about the CSV to the system prompt"""
    csv = ctx.deps.csv_data
    columns = csv.column_names()
    rows = csv.row_count()
    
    # Get a sample of the data to include in the prompt
    sample = csv.get_sample(3)
    
    return (
        f"The CSV has {rows} rows and {len(columns)} columns.\n"
        f"Columns: {', '.join(columns)}\n"
    )


@csv_analyzer_agent.tool
async def column_info(ctx: RunContext[CSVAnalyzerDependencies]) -> str:
    """Get information about the columns in the CSV"""
    csv = ctx.deps.csv_data
    columns = csv.column_names()
    return f"The CSV has {len(columns)} columns: {', '.join(columns)}"


@csv_analyzer_agent.tool
async def row_count(ctx: RunContext[CSVAnalyzerDependencies]) -> str:
    """Get the number of rows in the CSV"""
    csv = ctx.deps.csv_data
    return f"The CSV has {csv.row_count()} rows"


@csv_analyzer_agent.tool
async def column_stats(
    ctx: RunContext[CSVAnalyzerDependencies], column_name: str
) -> str:
    """Get statistical information about a numerical column"""
    try:
        csv = ctx.deps.csv_data
        stats = csv.get_stats(column_name)
        return (
            f"Statistics for column '{column_name}':\n"
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}\n"
            f"Min: {stats['min']:.2f}\n"
            f"Max: {stats['max']:.2f}\n"
            f"Standard Deviation: {stats['std']:.2f}"
        )
    except ValueError as e:
        return f"Error: {str(e)}"
    except TypeError as e:
        return f"Error: {str(e)}"


@csv_analyzer_agent.tool
async def unique_values(
    ctx: RunContext[CSVAnalyzerDependencies], column_name: str
) -> str:
    """Get unique values in a column"""
    try:
        csv = ctx.deps.csv_data
        values = csv.get_unique_values(column_name)
        if len(values) > 10:
            return f"Column '{column_name}' has {len(values)} unique values. First 10: {values[:10]}"
        else:
            return f"Unique values in column '{column_name}': {values}"
    except ValueError as e:
        return f"Error: {str(e)}"


@csv_analyzer_agent.tool
async def value_counts(
    ctx: RunContext[CSVAnalyzerDependencies], column_name: str
) -> str:
    """Get counts of unique values in a column"""
    try:
        csv = ctx.deps.csv_data
        counts = csv.get_value_counts(column_name)
        
        # Format output for readability
        counts_str = "\n".join([f"{k}: {v}" for k, v in list(counts.items())[:10]])
        
        if len(counts) > 10:
            return f"Value counts for column '{column_name}' (showing top 10 of {len(counts)}):\n{counts_str}"
        else:
            return f"Value counts for column '{column_name}':\n{counts_str}"
    except ValueError as e:
        return f"Error: {str(e)}"


@csv_analyzer_agent.tool
async def correlation_analysis(
    ctx: RunContext[CSVAnalyzerDependencies], column1: str, column2: str
) -> str:
    """Calculate correlation between two numerical columns"""
    try:
        csv = ctx.deps.csv_data
        corr = csv.correlation(column1, column2)
        
        # Interpret the correlation
        interpretation = ""
        if abs(corr) < 0.3:
            interpretation = "This indicates a weak correlation."
        elif abs(corr) < 0.7:
            interpretation = "This indicates a moderate correlation."
        else:
            interpretation = "This indicates a strong correlation."
            
        return (
            f"Correlation between '{column1}' and '{column2}': {corr:.4f}\n"
            f"{interpretation}"
        )
    except (ValueError, TypeError) as e:
        return f"Error: {str(e)}"


@csv_analyzer_agent.tool
async def group_by_analysis(
    ctx: RunContext[CSVAnalyzerDependencies], group_column: str, value_column: str, agg_func: str = 'mean'
) -> str:
    """Analyze data by grouping on one column and aggregating another"""
    try:
        csv = ctx.deps.csv_data
        grouped_data = csv.group_by_stats(group_column, value_column, agg_func)
        
        # Format output
        result = f"{agg_func.capitalize()} of '{value_column}' grouped by '{group_column}':\n"
        for group, value in grouped_data.items():
            result += f"{group}: {value:.2f}\n"
            
        return result
    except (ValueError, TypeError) as e:
        return f"Error: {str(e)}"


@csv_analyzer_agent.tool
async def time_series_trend(
    ctx: RunContext[CSVAnalyzerDependencies], date_column: str, value_column: str, freq: str = 'M'
) -> str:
    """Analyze time series data with specified frequency"""
    try:
        csv = ctx.deps.csv_data
        time_data = csv.time_series_analysis(date_column, value_column, freq)
        
        # Format output
        freq_map = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}
        freq_name = freq_map.get(freq, freq)
        
        result = f"{freq_name} trend of '{value_column}':\n"
        for date, value in list(time_data.items())[:10]:  # Limit to first 10 periods
            result += f"{date}: {value:.2f}\n"
            
        if len(time_data) > 10:
            result += f"... and {len(time_data) - 10} more periods"
            
        return result
    except (ValueError, TypeError) as e:
        return f"Error: {str(e)}"


@csv_analyzer_agent.tool
async def get_data_sample(
    ctx: RunContext[CSVAnalyzerDependencies], rows: int = 5
) -> str:
    """Get a sample of rows from the dataset"""
    try:
        csv = ctx.deps.csv_data
        sample = csv.get_sample(rows)
        
        # Format the sample as a table-like string
        columns = list(sample.keys())
        result = "Data sample:\n"
        
        # Add header
        result += " | ".join(columns) + "\n"
        result += "-" * (sum(len(col) for col in columns) + 3 * (len(columns) - 1)) + "\n"
        
        # Add rows
        for i in range(min(rows, len(next(iter(sample.values()))))):
            row = []
            for col in columns:
                val = sample[col][i]
                row.append(str(val))
            result += " | ".join(row) + "\n"
            
        return result
    except Exception as e:
        return f"Error retrieving sample: {str(e)}"


@csv_analyzer_agent.tool
async def filtered_analysis(
    ctx: RunContext[CSVAnalyzerDependencies], column: str, value: str, operator: str = '=='
) -> str:
    """Filter the data and return basic statistics"""
    try:
        csv = ctx.deps.csv_data
        
        # Handle type conversion for the value
        try:
            # Try numeric conversion first
            numeric_value = float(value)
            filtered_data = csv.filter_data(column, numeric_value, operator)
        except ValueError:
            # If not numeric, use as string
            filtered_data = csv.filter_data(column, value, operator)
        
        row_count = filtered_data.row_count()
        
        result = f"Filtered data where {column} {operator} {value}:\n"
        result += f"Number of rows: {row_count}\n"
        
        # Get numeric columns for summary statistics
        numeric_columns = [col for col in filtered_data.column_names() 
                         if pd.api.types.is_numeric_dtype(filtered_data.df[col])]
        
        if row_count > 0 and numeric_columns:
            result += "\nSummary statistics for numeric columns:\n"
            for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                stats = filtered_data.get_stats(col)
                result += f"{col} - Mean: {stats['mean']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}\n"
                
        return result
    except (ValueError, TypeError) as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    # Create test data
    csv_data = CSVData()  # Uses the built-in test dataframe
    deps = CSVAnalyzerDependencies(csv_data=csv_data)
    
    # Example queries
    queries = [
        "What columns are in the dataset?",
        # "Show me statistics for the Sales column",
        # "What's the correlation between Sales and Quantity?",
        # "What are the average Sales by Region?",
        # "Show me the monthly trend of Sales",
        # "Show me a sample of 3 rows from the data",
        "How many  products are there?"
    ]
    
    # Run queries
    for i, query in enumerate(queries, 1):
        result = csv_analyzer_agent.run_sync(query, deps=deps)
        print(f"\nQuery {i}: {query}")
        print(f"Result: {result.data.answer}\n")
        print("-" * 50)