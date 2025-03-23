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
# 1. Setup Logging
# --------------------------------------------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger("csv_analyzer")

# Set up the OpenAI model with Ollama as the provider
model = OpenAIModel(
    model_name='llama3.2:3b',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)
logger.info("Model initialized")

# --------------------------------------------------------------
# 2. Define Pydantic Models for Structure
# --------------------------------------------------------------

class ColumnStatistics(BaseModel):
    """Statistics for a single column."""
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unique_values: Optional[int] = None
    missing_values: Optional[int] = None
    
class CorrelationResult(BaseModel):
    """Result of correlation analysis."""
    correlation_matrix: Dict[str, Dict[str, float]]
    strongest_correlation: Tuple[str, str, float]
    weakest_correlation: Tuple[str, str, float]

class QueryModel(BaseModel):
    """Structure for query responses."""
    question: str
    answer: str = Field(description="The answer to the user's question based on the data")
    code_used: Optional[str] = Field(description="Python code used to generate the answer, if applicable")

class DataframeContext(BaseModel):
    """Structure to hold complete dataframe information."""
    dataframe_str: str
    dataframe_info: Dict[str, Any]
    columns: List[str]
    rows_count: int
    dtypes: Dict[str, str]

# --------------------------------------------------------------
# 3. Define Agent with Context and Structure
# --------------------------------------------------------------

csv_agent = Agent(
    model=model,
    result_type=QueryModel,
    deps_type=DataframeContext,
    retries = 3,
    system_prompt=(
        "You are a data analysis assistant specialized in pandas and numpy operations. "
        "Your job is to answer questions about CSV data and perform analysis. "
        "You have access to the complete dataset and a set of tools to analyze it. "
        "When using tools, make sure to reference the correct column names. "
        "Include the code you used in your answers when appropriate. "
        "Provide clear, accurate, and helpful responses based on the data provided."
    ),
)
logger.info("CSV Agent created")

# Global variable to store the current dataframe
current_df = None

# --------------------------------------------------------------
# 4. Define Data Analysis Tools
# --------------------------------------------------------------

@csv_agent.tool_plain
def get_column_names() -> List[str]:
    """Get the list of column names in the dataframe."""
    logger.info("Tool called: get_column_names")
    if current_df is None:
        return []
    return current_df.columns.tolist()

@csv_agent.tool_plain
def get_row_count() -> int:
    """Get the number of rows in the dataframe."""
    logger.info("Tool called: get_row_count")
    if current_df is None:
        return 0
    return len(current_df)

@csv_agent.tool_plain
def get_column_stats(column_name: str) -> ColumnStatistics:
    """Get detailed statistics for a specified column."""
    logger.info(f"Tool called: get_column_stats for column: {column_name}")
    if current_df is None or column_name not in current_df.columns:
        return ColumnStatistics()
    
    try:
        col = current_df[column_name]
        stats = ColumnStatistics(
            mean=float(col.mean()) if pd.api.types.is_numeric_dtype(col) else None,
            median=float(col.median()) if pd.api.types.is_numeric_dtype(col) else None,
            std=float(col.std()) if pd.api.types.is_numeric_dtype(col) else None,
            min=float(col.min()) if pd.api.types.is_numeric_dtype(col) else None,
            max=float(col.max()) if pd.api.types.is_numeric_dtype(col) else None,
            unique_values=col.nunique(),
            missing_values=col.isna().sum()
        )
        return stats
    except Exception as e:
        logger.error(f"Error in get_column_stats: {e}")
        return ColumnStatistics()

@csv_agent.tool_plain
def get_data_sample(n: int = 5) -> str:
    """Get a sample of n rows from the dataframe."""
    logger.info(f"Tool called: get_data_sample with n={n}")
    if current_df is None:
        return "No data available"
    return current_df.head(n).to_string()

@csv_agent.tool_plain
def get_correlation_matrix(columns: Optional[List[str]] = None) -> CorrelationResult:
    """
    Get correlation matrix for numeric columns.
    Optionally specify a subset of columns to analyze.
    """
    logger.info(f"Tool called: get_correlation_matrix for columns: {columns}")
    if current_df is None:
        return CorrelationResult(
            correlation_matrix={},
            strongest_correlation=("", "", 0.0),
            weakest_correlation=("", "", 0.0)
        )
    
    try:
        # Filter for numeric columns
        numeric_df = current_df.select_dtypes(include=['number'])
        
        # Filter for requested columns if specified
        if columns:
            valid_columns = [col for col in columns if col in numeric_df.columns]
            if not valid_columns:
                return CorrelationResult(
                    correlation_matrix={},
                    strongest_correlation=("", "", 0.0),
                    weakest_correlation=("", "", 0.0)
                )
            numeric_df = numeric_df[valid_columns]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().to_dict()
        
        # Find strongest and weakest correlations
        strongest = ("", "", 0.0)
        weakest = ("", "", 1.0)
        
        for col1 in corr_matrix:
            for col2 in corr_matrix[col1]:
                if col1 != col2:
                    corr_val = abs(corr_matrix[col1][col2])
                    if corr_val > strongest[2]:
                        strongest = (col1, col2, corr_matrix[col1][col2])
                    if corr_val < weakest[2]:
                        weakest = (col1, col2, corr_matrix[col1][col2])
        
        return CorrelationResult(
            correlation_matrix=corr_matrix,
            strongest_correlation=strongest,
            weakest_correlation=weakest
        )
    except Exception as e:
        logger.error(f"Error in get_correlation_matrix: {e}")
        return CorrelationResult(
            correlation_matrix={},
            strongest_correlation=("", "", 0.0),
            weakest_correlation=("", "", 0.0)
        )

@csv_agent.tool_plain
def run_custom_query(query_code: str) -> str:
    """
    Run custom pandas/numpy code on the dataframe.
    The dataframe is available as 'df' in the code.
    Returns the result as a string.
    """
    logger.info(f"Tool called: run_custom_query with code: {query_code}")
    if current_df is None:
        return "No data available"
    
    try:
        # Set up local variables for execution
        local_vars = {"df": current_df, "pd": pd, "np": np}
        
        # Execute the code
        exec(query_code, {}, local_vars)
        
        # Look for a result variable
        if 'result' in local_vars:
            result = local_vars['result']
            if isinstance(result, pd.DataFrame):
                return result.to_string()
            return str(result)
        
        return "Query executed, but no 'result' variable was defined"
    except Exception as e:
        logger.error(f"Error in run_custom_query: {e}")
        return f"Error executing query: {str(e)}"

# Add dynamic system prompt based on complete dataframe context
@csv_agent.system_prompt
async def add_dataframe_context(ctx: RunContext[DataframeContext]) -> str:
    logger.info("Adding dataframe context to system prompt")
    return (
        f"Dataset ({ctx.deps.rows_count} rows, {len(ctx.deps.columns)} columns):\n```\n{ctx.deps.dataframe_str}\n```\n"
        f"Available columns: {', '.join(ctx.deps.columns)}\n"
        f"Column data types: {ctx.deps.dtypes}\n"
        f"Dataset statistics summary: {ctx.deps.dataframe_info}"
    )

# --------------------------------------------------------------
# 5. CSV Processing Functions
# --------------------------------------------------------------

def validate_csv(file):
    """Validate and load a CSV file."""
    logger.info(f"Validating CSV file: {file.name}")
    try:
        df = pd.read_csv(file.name)
        logger.info(f"CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        global current_df
        current_df = df
        return df, "CSV file uploaded successfully!"
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None, f"Error: {str(e)}"

def get_dataframe_info(df):
    """Extract comprehensive dataframe information."""
    logger.info("Extracting dataframe information")
    # Get dataframe statistics
    stats = {}
    try:
        stats = df.describe().to_dict()
    except:
        # If describe fails (e.g., for non-numeric data)
        stats = {col: {"count": df[col].count()} for col in df.columns}
    
    # Get data types as strings
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
    
    return stats, dtypes

def answer_question(df, question):
    """Process a question using the PydanticAI agent with full dataframe context."""
    logger.info(f"Processing question: {question}")
    try:
        # Get comprehensive dataframe information
        df_info, dtypes = get_dataframe_info(df)
        
        # Create context for the agent with full dataframe
        context = DataframeContext(
            dataframe_str=df.head(10).to_string(),  # Send just a preview for context
            dataframe_info=df_info,
            columns=df.columns.tolist(),
            rows_count=len(df),
            dtypes=dtypes
        )
        
        # Run the agent with the question and context
        logger.info("Running agent with context")
        response = csv_agent.run_sync(
            user_prompt=f"Answer this question about the data: {question}",
            deps=context
        )
        
        logger.info("Agent response received")
        return response.data.answer, response.data.code_used
    except Exception as e:
        logger.error(f"Error in LLM processing: {e}")
        return f"Error in LLM processing: {e}", None

def plot_graph(df, x_column, y_column, plot_type="line"):
    """Generate a plot based on selected columns and plot type."""
    logger.info(f"Generating {plot_type} plot for {x_column} vs {y_column}")
    try:
        plt.figure()
        if plot_type == "line":
            plt.plot(df[x_column], df[y_column])
        elif plot_type == "bar":
            plt.bar(df[x_column], df[y_column])
        elif plot_type == "scatter":
            plt.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{plot_type.capitalize()} Plot: {x_column} vs {y_column}")
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return f"Error generating plot: {str(e)}"

# --------------------------------------------------------------
# 6. Main Processing Function
# --------------------------------------------------------------

def process_csv(file, question, x_column, y_column, plot_type):
    """Process CSV, answer questions, and generate plots."""
    logger.info("Processing CSV file and question")
    df, message = validate_csv(file)
    if df is None:
        logger.warning(f"CSV validation failed: {message}")
        return message, None, None

    # Answer the question using PydanticAI with full dataset
    answer, code_used = answer_question(df, question)
    
    # Generate the plot if columns are selected
    plot = None
    if x_column and y_column:
        logger.info(f"Generating plot with {x_column} and {y_column}")
        plot = plot_graph(df, x_column, y_column, plot_type)

    # Return results
    return answer, plot if isinstance(plot, plt.Figure) else None, code_used

# --------------------------------------------------------------
# 7. Gradio Interface
# --------------------------------------------------------------

with gr.Blocks() as app:
    gr.Markdown("# CSV Question Answering and Visualization with PydanticAI")

    with gr.Row():
        file_input = gr.File(label="Upload CSV File")
        question_input = gr.Textbox(label="Ask a Question", placeholder="E.g., What are the average values for each column?")

    with gr.Row():
        x_column = gr.Dropdown(label="X-Axis Column", choices=[])
        y_column = gr.Dropdown(label="Y-Axis Column", choices=[])
        plot_type = gr.Radio(choices=["line", "bar", "scatter"], label="Plot Type", value="line")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer")
        plot_output = gr.Plot(label="Graph")
    
    with gr.Row():
        code_output = gr.Code(language="python", label="Code Used")

    submit_button = gr.Button("Submit")

    # Update column dropdowns based on uploaded CSV
    def update_columns(file):
        logger.info("Updating column dropdowns")
        df, _ = validate_csv(file)
        if df is not None:
            columns = df.columns.tolist()
            return gr.Dropdown(choices=columns), gr.Dropdown(choices=columns)
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

    file_input.change(update_columns, inputs=file_input, outputs=[x_column, y_column])

    # Process inputs on button click
    submit_button.click(
        process_csv,
        inputs=[file_input, question_input, x_column, y_column, plot_type],
        outputs=[answer_output, plot_output, code_output]
    )

# Launch the app
if __name__ == "__main__":
    logger.info("Starting the application")
    app.launch()
    logger.info("Application closed")