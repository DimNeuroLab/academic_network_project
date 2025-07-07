import os
import json
import re
from collections import defaultdict
from pdf2image import convert_from_path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet

# Folder containing the experiment folders
base_folder = "/home/sabrina/academic_network_project/anp_models"

date_lower = "2024_04_24"
date_upper = "2024_11_16"
lr_lower = 0.00001
lr_upper = 0.00001

infosphere_string = ["Baseline (No infosphere)", "Future infosphere", "Infosphere TOP PAPER", "Infosphere TOP PAPER PER TOPIC"]
# Convert date strings to integers for comparison
date_lower_int = int(date_lower.replace("_", ""))
date_upper_int = int(date_upper.replace("_", ""))

# Dictionary to organize the results
results = [[{}, {}, {}, {}], [{}, {}, {}, {}]]

# Function to read the info.json file
def read_info_json(folder):
    info_file = os.path.join(folder, "info.json")
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = json.load(f)
            if info_data['infosphere_type'] == 0:
                del info_data['infosphere_parameters']
            return info_data
    return None

# Function to read the *_log.json file
def read_log_json(folder):
    log_file = None
    for file_name in os.listdir(folder):
        if file_name.endswith("_log.json"):
            log_file = os.path.join(folder, file_name)
            break
    if log_file and os.path.exists(log_file):
        with open(log_file) as f:
            log_data = json.load(f)
            return log_data
    return None

# Regex for the folder format
folder_pattern = re.compile(r'anp_link_prediction_co_author_(.*?)_(\d{4}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}')

# Scan all the experiment folders
for experiment_folder in os.listdir(base_folder):
    experiment_path = os.path.join(base_folder, experiment_folder)
    if not os.path.isdir(experiment_path):
        continue

    # Check if the folder name matches the desired format and falls within the date range
    match = folder_pattern.match(experiment_folder)
    if not match:
        continue

    folder_date = match.group(2)
    if not (date_lower_int <= int(folder_date.replace("_", "")) <= date_upper_int):
        continue

    # Read info.json
    info = read_info_json(experiment_path)
    if info is None:
        continue

    # Read *_log.json
    log_data = read_log_json(experiment_path)
    if log_data is None:
        continue

    # Extract the values of interest
    prediction_type = info.get("only_new", "N/A")
    lr = info.get("lr", "N/A")

    if not (lr_lower <= float(lr) <= lr_upper):
        continue

    # Validation accuracy, highest accuracy, and average of last 10 epochs using the log file data
    validation_accuracy = log_data["validation_accuracy_list"][-1] if log_data["validation_accuracy_list"] else "N/A"
    highest_accuracy = max(log_data["validation_accuracy_list"]) if log_data["validation_accuracy_list"] else "N/A"
    avg_last_10_epochs = sum(log_data["validation_accuracy_list"][-10:]) / 10 if len(log_data["validation_accuracy_list"]) >= 10 else "N/A"
    num_epochs = len(log_data["validation_accuracy_list"])
    edge_number = info.get("edge_number", "N/A")
    aggregation_type = info.get("aggregation_type", "N/A")
    drop_percentage = info.get("drop_percentage", "N/A")
    if "hgt" in experiment_folder or "htg" in experiment_folder:
        network_type = "HGT"
    else:
        network_type = "SAGE + to_hetero"

    # Create a title for the report
    folder = f"Folder: {experiment_folder}"

    # Create the body of the report
    body = f"Learning Rate: {lr},\n"
    body += f"Last Validation Accuracy: {validation_accuracy},\n"
    body += f"Highest Accuracy: {highest_accuracy},\n"
    body += f"Average of Last 10 Epochs: {avg_last_10_epochs},\n"
    body += f"Number of Epochs: {num_epochs},\n"
    body += f"Edge Number: {edge_number},\n"
    body += f"Aggregation Type: {aggregation_type},\n"
    body += f"Drop Percentage: {drop_percentage},\n"
    body += f"Network Type: {network_type}\n"

    infosphere_parameters = info.get("infosphere_parameters", None)

    # Save the images if they exist
    images = []
    for base_name in ["accuracy", "CM", "loss"]:
        image_path = ""
        list_ex = os.listdir(experiment_path)
        for file_name in list_ex:
            if base_name in file_name:
                # Construct the full path of the file
                image_path = os.path.join(experiment_path, file_name)
                break
        if os.path.exists(image_path):
            # Convert the PDF to a JPEG image
            pdf_images = convert_from_path(image_path, dpi=300, fmt='jpeg')
            for idx, pdf_image in enumerate(pdf_images):
                # Save the temporary image
                temp_image_path = f"temp_{file_name}_{experiment_folder}.jpeg"
                pdf_image.save(temp_image_path, "JPEG")
                images.append(temp_image_path)

    # Add the data to the results dictionary
    infotype = info["infosphere_type"] if info.get("infosphere_type") else 0
    if not results[prediction_type][infotype].get(infosphere_parameters):
        results[prediction_type][infotype][infosphere_parameters] = []
    results[prediction_type][infotype][infosphere_parameters].append({
        "folder": folder,
        "body": body,
        "images": images,
        "learning_rate": lr
    })

# Function to create a PDF report
def create_pdf_report(only_new, infotype, entries, infosphere_string):
    doc_name = f"MIX_INFOSPHERE_report_{infosphere_string[infotype]}_{'only_new' if only_new else 'all_co_authors'}_{date_lower}_{date_upper}_{lr_lower}_{lr_upper}.pdf"
    doc_name = doc_name.replace(" ", "_")  # Replace spaces with underscores for file naming
    doc = SimpleDocTemplate(doc_name, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    # Header
    elements.append(Paragraph("Experiment Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Title for the specific report section
    elements.append(Paragraph("Only new" if only_new else "All co-authors", styles["Heading1"]))
    elements.append(Paragraph(f"{infosphere_string[infotype]}", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    # Sort entries by infosphere parameters
    sorted_entries2 = dict(sorted(entries.items(), key=lambda x: x[0]))

    for infosphere_parameter, entries_list in sorted_entries2.items():
        if infosphere_parameter:
            elements.append(Paragraph(f"Parameters: {infosphere_parameter}", styles["Heading3"]))
        entries_list.sort(key=lambda x: x.get("learning_rate", 0))
        for entry in entries_list:
            elements.append(Paragraph(entry["body"], styles["Normal"]))

            # List of images to insert in this section
            images_section = []

            # Add the images to the current section list
            for image_path in entry["images"]:
                image = PILImage.open(image_path)
                width, height = image.size
                max_width = 200  # Width in points
                aspect = height / width
                adjusted_width = max_width
                adjusted_height = adjusted_width * aspect
                image_obj = Image(image_path, width=adjusted_width, height=adjusted_height)
                images_section.append(image_obj)

            # Create a table to display the images horizontally
            if images_section:
                table_data = [images_section]
                table = Table(table_data, colWidths=len(images_section) * [200])
                elements.append(table)

            elements.append(Paragraph(entry["folder"], styles["Heading6"]))
            elements.append(Spacer(1, 12))

    doc.build(elements)

# Create separate PDF reports for each combination of only_new and infosphere_type
for only_new in range(2):
    for i, entries2 in enumerate(results[only_new]):
        if entries2:
            create_pdf_report(only_new, i, entries2, infosphere_string)

# Remove the temporary images
for only_new in range(2):
    for i, entries2 in enumerate(results[only_new]):
        for infosphere_parameter, entries in entries2.items():
            for entry in entries:
                for image_path in entry["images"]:
                    os.remove(image_path)

print("The reports were successfully created.")
