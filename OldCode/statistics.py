import sqlite3
import datetime
from tqdm import tqdm
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.io import output_file
from bokeh.models import HoverTool, ColumnDataSource
import requests

# Connect to SQLite database
conn = sqlite3.connect('./huggingface2.db')
cursor = conn.cursor()


#check how many are in transformers:
def is_model_available_in_transformers(model_name):

    url = f"https://huggingface.co/api/models/{model_name}"
    response = requests.get(url)

    if response.status_code == 200:
        model_data = response.json()
        # Check if the model supports the "transformers" library
        if "transformers" in model_data.get("library_name", ""):
            return 1
        else:

            return 0
# Example Usage



# Fetch data
cursor.execute("SELECT model_name FROM Models")
rows = cursor.fetchall()

# Extracting data
numberOfTransformerModels= 0


for model_name in tqdm(rows):
    numberOfTransformerModels+=is_model_available_in_transformers(model_name[0])

exit(0)

# Fetch data
cursor.execute("SELECT downloads, likes, lastModified FROM Models")
rows = cursor.fetchall()
conn.close()  # Close connection

# Extracting data
downloads = []
likes = []

for row in rows:
    downloads.append(row[0])
    likes.append(row[1])

# --- Bokeh Output ---
output_file("log_binned_histograms.html")  # Save as an HTML file

# Logarithmic binning function
# Logarithmic binning function (updated)
def create_log_binned_histogram(data, title, x_label, color="blue"):
    # Filter out zeros and replace them with a small value
    data = np.array(data)
    data = data[data > 0]  # Keep only positive values for log10

    # Define bins in log scale
    bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num=20)
    hist, edges = np.histogram(data, bins=bins)

    # Prepare data for Bokeh
    source = ColumnDataSource(data=dict(
        left=edges[:-1], right=edges[1:], count=hist
    ))

    # Create plot
    p = figure(title=title, background_fill_color="#fafafa", x_axis_type="log")
    p.quad(source=source, top="count", bottom=0, left="left", right="right",
           fill_color=color, line_color="black", alpha=0.7)

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [("Range", "@left{0,0} - @right{0,0}"), ("Count", "@count")]
    p.add_tools(hover)

    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = "Number of Models"
    return p


print(f'The number of Total elements is: {len(downloads)}')
down=np.array(downloads)
print(f'number of 0 downloads: {len(down[down<15])}')
l=np.array(likes)
print(f'number of 0 likes: {len(l[l<15])}')
# --- Create Histograms ---
p1 = create_log_binned_histogram(downloads, "Distribution of Model Downloads (Log Scale)", "Downloads", color="orange")
p2 = create_log_binned_histogram(likes, "Distribution of Model Likes (Log Scale)", "Likes", color="green")

# --- Show Both Plots in Column Layout ---
show(column(p1, p2))
