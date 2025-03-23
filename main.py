import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
# Set up the OpenAI model with Ollama as the provider
model = OpenAIModel(
    model_name='llama3.2:3b',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)

# --------------------------------------------------------------
# 1. Define Pydantic Models for Structure
# --------------------------------------------------------------

class QueryModel(BaseModel):
    """Structure for query responses."""
    question: str
    answer: str = Field(description="The answer to the user's question based on the data")

class DataframeContext(BaseModel):
    """Structure to hold complete dataframe information."""
    dataframe_str: str
    dataframe_info: Dict[str, Any]
    columns: list[str]
    rows_count: int
    dtypes: Dict[str, str]

# --------------------------------------------------------------
# 2. Define Agent with Context and Structure
# --------------------------------------------------------------

csv_agent = Agent(
    model=model,
    result_type=QueryModel,
    deps_type=DataframeContext,
    system_prompt= """
You are a business analyst. Your responses must be clear, concise, and directly relevant to the question. 
Avoid unnecessary elaboration, opinions, or redundant details. 
Focus on actionable insights, key takeaways, and essential data. Use bullet points where applicable.
""")

# Add dynamic system prompt based on complete dataframe context
@csv_agent.system_prompt
async def add_dataframe_context(ctx: RunContext[DataframeContext]) -> str:
    return (
        f"Dataset ({ctx.deps.rows_count} rows, {len(ctx.deps.columns)} columns):\n```\n{ctx.deps.dataframe_str}\n```\n"
        f"Available columns: {', '.join(ctx.deps.columns)}\n"
        f"Column data types: {ctx.deps.dtypes}\n"
        f"Dataset statistics: {ctx.deps.dataframe_info}"
    )

# --------------------------------------------------------------
# 3. CSV Processing Functions
# --------------------------------------------------------------

def validate_csv(file):
    """Validate and load a CSV file."""
    try:
        df = pd.read_csv(file.name)
        return df, "CSV file uploaded successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_dataframe_info(df):
    """Extract comprehensive dataframe information."""
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
    try:
        # Get comprehensive dataframe information
        df_info, dtypes = get_dataframe_info(df)
        
        # Create context for the agent with full dataframe
        context = DataframeContext(
            dataframe_str=df.to_string(),
            dataframe_info=df_info,
            columns=df.columns.tolist(),
            rows_count=len(df),
            dtypes=dtypes
        )
        
        # Run the agent with the question and context
        response = csv_agent.run_sync(
            user_prompt=f"Answer this question about the data: {question}",
            deps=context
        )
        
        return response.data.answer
    except Exception as e:
        return f"Error in LLM processing: {e}"

def plot_graph(df, x_column, y_column, plot_type="line"):
    """Generate a plot based on selected columns and plot type."""
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
        return f"Error generating plot: {str(e)}"

# --------------------------------------------------------------
# 4. Main Processing Function
# --------------------------------------------------------------

def process_csv(file, question, x_column, y_column, plot_type):
    """Process CSV, answer questions, and generate plots."""
    df, message = validate_csv(file)
    if df is None:
        return message, None

    # Answer the question using PydanticAI with full dataset
    answer = answer_question(df, question)

    # Generate the plot if columns are selected
    plot = None
    if x_column and y_column:
        plot = plot_graph(df, x_column, y_column, plot_type)

    # Return results
    return answer, plot if isinstance(plot, plt.Figure) else None

# --------------------------------------------------------------
# 5. Gradio Interface
# --------------------------------------------------------------

with gr.Blocks() as app:
    gr.Markdown("# CSV Question Answering and Visualization with PydanticAI")

    with gr.Row():
        file_input = gr.File(label="Upload CSV File")
        question_input = gr.Textbox(label="Ask a Question")

    with gr.Row():
        x_column = gr.Dropdown(label="X-Axis Column", choices=[])
        y_column = gr.Dropdown(label="Y-Axis Column", choices=[])
        plot_type = gr.Radio(choices=["line", "bar", "scatter"], label="Plot Type", value="line")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer")
        plot_output = gr.Plot(label="Graph")

    submit_button = gr.Button("Submit")

    # Update column dropdowns based on uploaded CSV
    def update_columns(file):
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
        outputs=[answer_output, plot_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()