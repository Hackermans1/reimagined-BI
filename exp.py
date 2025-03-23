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
    
    def __init__(self, file_path: str):
        """Initialize with the path to a CSV file"""
        self.file_path = file_path
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded CSV file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise ValueError(f"Failed to load CSV file: {e}")
    
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
    # sample_str = "\n".join([f"{col}: {sample[col]}" for col in columns])
    
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


if __name__ == '__main__':
    # Example usage
    csv_path = r"C:\Users\sudar\Downloads\sales_data.csv"
    csv_data = CSVData(csv_path)
    deps = CSVAnalyzerDependencies(csv_data=csv_data)
    
    # Example 1: Basic information query
    result = csv_analyzer_agent.run_sync("give the columns of the data?", deps=deps)
    print("Query 1 result:", result.data)
    
    # # # Example 2: Statistical query
    # result = csv_analyzer_agent.run_sync(
    #     "What are the statistics for the 'sales' column?", 
    #     deps=deps
    # )
    # print("Query 2 result:", result.data)
    
    # # Example 3: Correlation query
    # result = csv_analyzer_agent.run_sync(
    #     "unique values in Product ?", 
    #     deps=deps
    # )
    # print("Query 3 result:", result.data)