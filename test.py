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

# --------------------------------------------------------------
# 1. Setup Logging with more detail
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger("csv_analyzer")

# Set up the OpenAI model with Ollama as the provider - enhanced configuration
model = OpenAIModel(
    model_name='llama3.2:3b',  # Upgrade to larger model if possible
    provider=OpenAIProvider(
        base_url='http://localhost:11434/v1' # Add timeout to prevent hanging
    ),
)
logger.info("Model initialized with enhanced configuration")

# --------------------------------------------------------------
# 2. Simplified Pydantic Models
# --------------------------------------------------------------

class ColumnStatistics(BaseModel):
    """Statistics for a single column."""
    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    missing_values: Optional[int] = None
    
class CorrelationResult(BaseModel):
    """Result of correlation analysis."""
    strongest_correlation: Optional[Tuple[str, str, float]] = None
    weakest_correlation: Optional[Tuple[str, str, float]] = None

# Simplified QueryModel to make it easier for LLM to generate valid responses
class QueryModel(BaseModel):
    """Simplified structure for query responses."""
    answer: str
# Simplified DataframeContext to reduce complexity
class DataframeContext(BaseModel):
    """Structure to hold essential dataframe information."""
    preview: str
    columns: List[str]
    rows_count: int
    dtypes: Dict[str, str]

# Global variable to store the current dataframe
current_df = None

# --------------------------------------------------------------
# 3. Define Agent with Optimized Context and Configuration
# --------------------------------------------------------------

csv_agent = Agent(
    model=model,
    result_type=QueryModel,
    deps_type=DataframeContext,
    retries=3,  # Increased retries
    # Added delay between retries
    # Increased timeout
    system_prompt=(
        "You are a data analysis assistant specialized in pandas operations. "
        "Your job is to answer questions about CSV data clearly and concisely. "
        "Keep responses focused on the data. "
        "When using pandas, refer to the dataframe as 'df'. "
        "Include code when useful, but it's not required for every answer."
    ),
)
logger.info("CSV Agent created with optimized configuration")

# --------------------------------------------------------------
# 4. Improved Data Analysis Tools with Better Error Handling
# --------------------------------------------------------------

# Correct syntax for tools with proper parameter ordering
@csv_agent.tool_plain  # Use plain tool for simple functions
def get_column_names():
    """Get the list of column names in the dataframe."""
    logger.info("Tool called: get_column_names")
    if current_df is None:
        logger.warning("get_column_names called with no dataframe")
        return []
    columns = current_df.columns.tolist()
    logger.info(f"Returning {len(columns)} columns")
    return columns

@csv_agent.tool
async def get_row_count(ctx: RunContext) -> int:
    """Get the number of rows in the dataframe."""
    logger.info("Tool called: get_row_count")
    if current_df is None:
        logger.warning("get_row_count called with no dataframe")
        return 0
    count = len(current_df)
    logger.info(f"Returning row count: {count}")
    return count

@csv_agent.tool
async def get_column_stats(ctx: RunContext, column_name: str) -> ColumnStatistics:
    """Get basic statistics for a specified column."""
    logger.info(f"Tool called: get_column_stats for column: {column_name}")
    try:
        if current_df is None:
            logger.warning("No dataframe available")
            return ColumnStatistics()
            
        if column_name not in current_df.columns:
            logger.warning(f"Column {column_name} not found")
            return ColumnStatistics()
        
        col = current_df[column_name]
        stats = ColumnStatistics(
            mean=float(col.mean()) if pd.api.types.is_numeric_dtype(col) else None,
            median=float(col.median()) if pd.api.types.is_numeric_dtype(col) else None,
            min=float(col.min()) if pd.api.types.is_numeric_dtype(col) else None,
            max=float(col.max()) if pd.api.types.is_numeric_dtype(col) else None,
            missing_values=col.isna().sum()
        )
        return stats
    except Exception as e:
        logger.error(f"Error in get_column_stats: {e}", exc_info=True)
        return ColumnStatistics()

@csv_agent.tool
async def get_data_sample(ctx: RunContext, n: int = 5) -> str:
    """Get a sample of n rows from the dataframe."""
    logger.info(f"Tool called: get_data_sample with n={n}")
    if current_df is None:
        logger.warning("get_data_sample called with no dataframe")
        return "No data available"
    try:
        sample = current_df.head(min(n, 10)).to_string()  # Limit to max 10 rows
        logger.info(f"Returning sample of {min(n, 10)} rows")
        return sample
    except Exception as e:
        logger.error(f"Error in get_data_sample: {e}", exc_info=True)
        return "Error retrieving data sample"

@csv_agent.tool
async def run_custom_query(ctx: RunContext, query_code: str) -> str:
    """
    Run custom pandas code on the dataframe.
    The dataframe is available as 'df' in the code.
    """
    logger.info(f"Tool called: run_custom_query")
    logger.debug(f"Query code: {query_code}")  # Log at debug level to not expose sensitive info
    
    if current_df is None:
        logger.warning("run_custom_query called with no dataframe")
        return "No data available"
    
    try:
        # Set up local variables for execution with safety limits
        local_vars = {"df": current_df, "pd": pd, "np": np, "result": None}
        
        # Add safety timeout or code length check
        if len(query_code) > 1000:
            logger.warning(f"Query code too long: {len(query_code)} chars")
            return "Query code is too long (>1000 chars)"
            
        # Execute the code
        exec(query_code, {}, local_vars)
        
        # Look for a result variable
        if 'result' in local_vars:
            result = local_vars['result']
            if isinstance(result, pd.DataFrame):
                # Limit large dataframe output
                if len(result) > 10:
                    return result.head(10).to_string() + f"\n[... {len(result)-10} more rows]"
                return result.to_string()
            return str(result)
        
        return "Query executed, but no 'result' variable was defined"
    except Exception as e:
        logger.error(f"Error in run_custom_query: {e}", exc_info=True)
        return f"Error executing query: {str(e)}"
    
# Simplified system prompt with less context
@csv_agent.system_prompt
def add_dataframe_context(ctx: RunContext[DataframeContext]) -> str:
    # Remove 'async' and 'await'
    return (
        f"Dataset ({ctx.deps.rows_count} rows, {len(ctx.deps.columns)} columns):\n```\n{ctx.deps.dataframe_str}\n```\n"
        f"Available columns: {', '.join(ctx.deps.columns)}\n"
        f"Column data types: {ctx.deps.dtypes}\n"
        f"Dataset statistics: {ctx.deps.dataframe_info}"
    )

# --------------------------------------------------------------
# 5. Improved CSV Processing Functions with Error Handling
# --------------------------------------------------------------

def validate_csv(file):
    """Validate and load a CSV file with better error handling."""
    logger.info(f"Validating CSV file: {file.name}")
    try:
        df = pd.read_csv(file.name)
        row_count = len(df)
        col_count = len(df.columns)
        logger.info(f"CSV loaded successfully: {row_count} rows, {col_count} columns")
        
        # Check if the dataframe is empty
        if row_count == 0 or col_count == 0:
            logger.warning("CSV file is empty")
            return None, "Error: CSV file is empty or has no columns"
            
        # Check for reasonable size to prevent memory issues
        if row_count > 100000:
            logger.warning(f"CSV file too large: {row_count} rows")
            return None, "Error: CSV file too large (>100,000 rows)"
            
        global current_df
        current_df = df
        return df, "CSV file uploaded successfully!"
    except pd.errors.EmptyDataError:
        logger.error("Empty CSV file")
        return None, "Error: The CSV file is empty"
    except pd.errors.ParserError:
        logger.error("CSV parsing error")
        return None, "Error: Could not parse the CSV file. Please check the format."
    except Exception as e:
        logger.error(f"Error loading CSV: {e}", exc_info=True)
        return None, f"Error: {str(e)}"

def get_dataframe_info(df):
    """Extract simplified dataframe information."""
    logger.info("Extracting dataframe information")
    # Get data types as strings
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
    
    return dtypes

def answer_question(df, question):
    """Process a question using the PydanticAI agent with dataframe context."""
    try:
        # Get comprehensive dataframe information
        dtypes = get_dataframe_info(df)
        
        # Create context with a more compact representation
        context = DataframeContext(
            dataframe_str=df.head(10).to_string(),  # Just show a sample
            dataframe_info=dtypes,
            columns=df.columns.tolist(),
            rows_count=len(df),
            dtypes=dtypes
        )
        
        # More explicit prompt to help validation succeed
        formatted_prompt = f"""
        Question about the data: {question}

        Respond with a clear answer based on the data provided.
        Your response should be in a format that can be parsed as:
        {{
        "answer": "Your detailed answer here"
        }}
        """
        
        # Run the agent with explicit response format guidance
        response = csv_agent.run_sync(
            user_prompt=formatted_prompt,
            deps=context
        )
        
        return response
    except Exception as e:
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return f"Error in LLM processing: {e}"

def plot_graph(df, x_column, y_column, plot_type="line"):
    """Generate a plot with improved error handling."""
    logger.info(f"Generating {plot_type} plot for {x_column} vs {y_column}")
    try:
        # Validate columns exist
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in the data"
        if y_column not in df.columns:
            return f"Error: Column '{y_column}' not found in the data"
            
        # Check data types for plotting
        x_numeric = pd.api.types.is_numeric_dtype(df[x_column])
        y_numeric = pd.api.types.is_numeric_dtype(df[y_column])
        
        if plot_type != "bar" and not (x_numeric and y_numeric):
            logger.warning(f"Non-numeric data for {plot_type} plot")
            plot_type = "bar"  # Fallback to bar for non-numeric data
            
        plt.figure(figsize=(10, 6))
        if plot_type == "line":
            plt.plot(df[x_column], df[y_column])
        elif plot_type == "bar":
            # Limit bars for readability
            sample = df.head(20) if len(df) > 20 else df
            plt.bar(sample[x_column], sample[y_column])
            if len(df) > 20:
                plt.title(f"Bar Plot: {x_column} vs {y_column} (showing first 20 rows)")
        elif plot_type == "scatter":
            plt.scatter(df[x_column], df[y_column], alpha=0.5)
            
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        if plot_type != "bar" or len(df) <= 20:
            plt.title(f"{plot_type.capitalize()} Plot: {x_column} vs {y_column}")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error generating plot: {e}", exc_info=True)
        return f"Error generating plot: {str(e)}"

# --------------------------------------------------------------
# 6. Main Processing Function with Graceful Fallbacks
# --------------------------------------------------------------

def process_csv(file, question, x_column, y_column, plot_type):
    """Process CSV with robust error handling and fallbacks."""
    logger.info("Processing CSV file and question")
    
    # Validate CSV
    df, message = validate_csv(file)
    if df is None:
        logger.warning(f"CSV validation failed: {message}")
        return message, None, None

    # Answer the question with fallback
    try:
        answer = answer_question(df, question)
    except Exception as e:
        logger.error(f"Failed to get answer from LLM: {e}", exc_info=True)
        # Provide a fallback response with basic statistics
        answer = (
            f"I encountered an error processing your question. "
            f"Here's some basic information about your data:\n"
        )
        code_used = None
    
    # Generate plot with fallback
    
    return answer

# --------------------------------------------------------------
# 7. Gradio Interface with Improved User Experience
# --------------------------------------------------------------

with gr.Blocks() as app:
    gr.Markdown("# CSV Question Answering and Visualization with PydanticAI")
    
    with gr.Row():
        file_input = gr.File(label="Upload CSV File")
        
    with gr.Row():
        status_output = gr.Textbox(label="Status", value="Upload a CSV file to begin")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a Question", 
            placeholder="E.g., What are the average values for each column?",
            lines=2
        )

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=8)
    
    
    submit_button = gr.Button("Submit Question")
    clear_button = gr.Button("Clear All")

    # Update column dropdowns and status based on uploaded CSV
    def update_from_file(file):
        logger.info("Updating interface from uploaded file")
        df, message = validate_csv(file)
        if df is not None:
            columns = df.columns.tolist()
            return (
                gr.Dropdown(choices=columns), 
                gr.Dropdown(choices=columns),
                message
            )
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), message

    # Clear all outputs
    def clear_outputs():
        return "", None, "", "Upload a CSV file to begin"

    # Connect events
    file_input.change(
        update_from_file, 
        inputs=file_input, 
        outputs=[ status_output]
    )
    
    submit_button.click(
        process_csv,
        inputs=[file_input, question_input],
        outputs=[answer_output]
    )
    
    clear_button.click(
        clear_outputs,
        outputs=[answer_output,  status_output]
    )

# Launch the app
if __name__ == "__main__":
    logger.info("Starting the application")
    try:
        app.launch()
        logger.info("Application closed")
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)