import numpy as np
import pandas as pd
import math
import os
from Parameter import args, dataset_config, PROJECT_ROOT
from workflow_utils import read_optional_label_fasta, finalize_features, resolve_project_path
from sequence_feature_common import output_name


# Calculate amino acid conservation.
def extract_section(file_path):
    found_start = False
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                found_start = True
            elif line.startswith('//'):
                break
            elif found_start:
                result.append(line.strip())
    return result


def calculate_value(sij, pib):
    if sij != '*':
        sij = float(sij)
        sij = 2 ** (-1 * sij / 1000)
        # Calculate pi * log2(pi / pib).
        return sij * math.log2(sij / pib)
    else:
        return 0


def calculate_relative_entropy_from_hhm(file_path, amino_acid_background):
    # print(f"Processing file: {file_path}")
    lines = extract_section(file_path)
    selected_lines = lines[4:][::3]  # Starting from the fifth line, keep every third line.

    # Parse the data and create a DataFrame.
    df = pd.DataFrame([line.split() for line in selected_lines])

    # Swap the first and second columns.
    df.iloc[:, [0, 1]] = df.iloc[:, [1, 0]].to_numpy()

    # Remove the last column.
    df = df.iloc[:, :-1]

    # Set column names.
    df.columns = ['index', 'AA'] + list(amino_acid_background.keys())

    # Calculate the relative entropy contribution for each amino acid.
    for col_index in range(2, df.shape[1]):
        amino_acid = df.columns[col_index]
        pib = amino_acid_background[amino_acid]
        # df.iloc[:, col_index] = df.iloc[:, col_index].apply(lambda sij: calculate_value(sij, pib))
        df[df.columns[col_index]] = df[df.columns[col_index]].apply(lambda sij: calculate_value(sij, pib))

    # Calculate relative entropy.
    df['relative_entropy'] = df.iloc[:, 2:].sum(axis=1)

    return df[['index', 'relative_entropy']]  # Keep only the index and relative entropy.


def merge_relative_entropy_data(input_folder):
    all_column_data = {}
    max_length = 0

    # Background frequencies for the 20 amino acids.
    amino_acid_background = {
        "A": 0.0834, "C": 0.0157, "D": 0.0541, "E": 0.061, "F": 0.0384,
        "G": 0.0668, "H": 0.0228, "I": 0.0533, "K": 0.052, "L": 0.0925,
        "M": 0.0218, "N": 0.0428, "P": 0.0523, "Q": 0.0401, "R": 0.0598,
        "S": 0.079, "T": 0.0583, "V": 0.0631, "W": 0.0127, "Y": 0.0287
    }

    # Process all .hhm files.
    for file_name in filter(lambda f: f.endswith('.hhm'), os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file_name)
        df = calculate_relative_entropy_from_hhm(file_path, amino_acid_background)

        column_name = file_name.split('.')[0]  # Use the filename prefix as the column name.
        all_column_data[column_name] = df['relative_entropy'].tolist()

        max_length = max(max_length, len(all_column_data[column_name]))  # Track the longest column.

    # Pad shorter columns to a common length.
    for column_name in all_column_data:
        current_length = len(all_column_data[column_name])
        if current_length < max_length:
            # all_column_data[column_name] += [0] * (max_length - current_length)
            all_column_data[column_name] += [np.nan] * (max_length - current_length)

    # Create the merged DataFrame.
    result_df = pd.DataFrame(all_column_data)

    return result_df

# Calculate flexible-linker segment conservation features.
# Process an .apod file and extract the required information.
def extract_info_from_apod(apod_file_path):
    with open(apod_file_path, 'r') as file:
        lines = [line.rstrip('\n') for line in file]  # Preserve content except the trailing newline.

    extracted_info = []
    current_accession = None
    current_data_lines = []  # Data lines for the current record.

    for line in lines:
        if line.startswith('>'):
            # When a new record starts, process the previous record first.
            if current_accession is not None:
                # Ensure that there are enough data lines, at least two.
                if len(current_data_lines) >= 2:
                    binary_value = current_data_lines[1]
                    extracted_info.append((current_accession, binary_value))

            # Initialize a new record.
            current_accession = line[1:].strip()  # Remove the '>' marker and surrounding whitespace.
            current_data_lines = []
        else:
            # Lines not starting with '>' are treated as data lines, including empty lines.
            current_data_lines.append(line.strip())

    # Process the last record.
    if current_accession is not None and len(current_data_lines) >= 2:
        binary_value = current_data_lines[1]
        extracted_info.append((current_accession, binary_value))

    return extracted_info


# Convert all 1-segments shorter than 4 residues to 0.
def replace_short_segments(sequence):
    new_sequence = ''
    current_length = 0

    for char in sequence:
        if char == '1':
            current_length += 1
        else:
            if current_length > 0:
                # At a non-'1' character, check whether the current segment is shorter than 4.
                if current_length < 4:
                    new_sequence += '0' * current_length  # Replace '1' in short segments with '0'.
                else:
                    new_sequence += '1' * current_length  # Keep long segments.
                current_length = 0
            new_sequence += char  # Append the non-'1' character.

    # Process the last segment.
    if current_length > 0:
        if current_length < 4:
            new_sequence += '0' * current_length  # Replace '1' in short segments with '0'.
        else:
            new_sequence += '1' * current_length  # Keep long segments.

    return new_sequence


# Process one .apod file and return the converted data.
def process_single_apod_file(apod_file_path):
    extracted_info = extract_info_from_apod(apod_file_path)  # Extract information.

    # Convert each extracted binary sequence.
    processed_data = []
    for accession_name, binary_value in extracted_info:
        new_binary_value = replace_short_segments(binary_value)  # Process the binary sequence.
        processed_data.append((accession_name, new_binary_value))  # Store the processed data.

    return processed_data


def processed_data_to_dataframe(processed_data):
    # Convert processed_data to a dictionary: accession_name as columns and binary values as data.
    data_dict = {}
    max_length = 0

    # Process each accession and its corresponding binary value.
    for accession_name, binary_value in processed_data:
        data_dict[accession_name] = list(binary_value)  # Convert the binary value to a list.
        max_length = max(max_length, len(binary_value))  # Track the longest binary sequence.

    # Pad shorter binary sequences to the same length.
    for accession_name in data_dict:
        current_length = len(data_dict[accession_name])
        if current_length < max_length:
            data_dict[accession_name] += [np.nan] * (max_length - current_length)  # Pad with NaN.

    # Create a DataFrame.
    df = pd.DataFrame(data_dict)
    df = df[sorted(df.columns, reverse=True)]

    return df


# Process disorder data.
def classify_amino_acids(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines.

    # Remove comment and header lines.
    lines = [line.strip() for line in lines if not line.startswith('#') and not line.startswith('Num')]

    sequence = ""
    classifications = []

    # Read the data and perform the initial classification.
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        score = float(parts[2])
        classification = 1 if score > 0.576 else 0  # Binary classification.
        classifications.append(classification)
        sequence += str(classification)

    # Handle continuous 1-segments shorter than 4 residues.
    processed_sequence = list(sequence)
    i = 0
    while i < len(processed_sequence):
        if processed_sequence[i] == '1':
            start = i
            while i < len(processed_sequence) and processed_sequence[i] == '1':
                i += 1
            end = i
            if end - start < 4:  # Convert segments shorter than 4 to 0.
                for j in range(start, end):
                    processed_sequence[j] = '0'
        else:
            i += 1

    # Convert the final classification result to an integer list.
    final_classifications = [int(c) for c in processed_sequence]

    return final_classifications


def process_folder(folder_path):
    all_data = {}  # Store classification data from all files.
    max_length = 0  # Track the longest classification sequence.

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            classifications = classify_amino_acids(file_path)

            column_name = os.path.splitext(filename)[0]  # Use the filename as the column name.
            all_data[column_name] = classifications

            max_length = max(max_length, len(classifications))  # Track the longest column length.

    # Pad shorter columns so that all columns have the same length.
    for column_name in all_data:
        current_length = len(all_data[column_name])
        if current_length < max_length:
            all_data[column_name] += [np.nan] * (max_length - current_length)  # Pad shorter columns with NaN.

    # Create a DataFrame.
    result_df = pd.DataFrame(all_data)

    return result_df


# Process secondary-structure data.
def extract_second_column_to_dataframe(root_folder):
    """
    Traverse subfolders under root_folder, extract the SS column from seq.phipsi,
    and merge the columns into one DataFrame using folder names as column names.
    Missing values are padded automatically.
    """
    data_frames = []  # Store all DataFrames.
    column_names = []  # Store column names, using folder names.

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file == "seq.phipsi":
                file_path = os.path.join(subdir, file)
                folder_name = os.path.basename(subdir)  # Get the folder name.

                try:
                    # Read the seq.phipsi file.
                    df = pd.read_csv(file_path, delim_whitespace=True, header=None, on_bad_lines="skip")

                    # Ensure that the file has enough columns.
                    if df.shape[1] > 1:
                        second_column = df.iloc[:, 2]  # Extract the SS column.
                        data_frames.append(pd.DataFrame(second_column))  # Convert to a DataFrame.
                        column_names.append(folder_name)  # Record the column name.

                    else:
                        print(f"File {file_path} has insufficient columns. Skipping.")

                except Exception as e:
                    print(f"Error while processing {file_path}: {e}")

    # Merge DataFrames by aligning to the maximum row count and padding with NaN.
    result_df = pd.concat(data_frames, axis=1, ignore_index=True)

    # Set column names, using folder names.
    result_df.columns = column_names

    return result_df


def process_ss_column(ss_list):
    """
    Process an SS column by replacing short continuous segments with NaN.
    H and C segments shorter than 3 are replaced; E segments shorter than 2 are replaced.
    Existing NaN values are preserved.
    """
    result = []
    i = 0
    while i < len(ss_list):
        # Skip NaN values.
        if pd.isna(ss_list[i]):
            result.append(ss_list[i])
            i += 1
            continue

        if ss_list[i] == 'H':
            start = i
            while i < len(ss_list) and ss_list[i] == 'H':
                i += 1
            # Replace continuous H segments shorter than 3 with NaN.
            if i - start < 3:
                result.extend([np.nan] * (i - start))
            else:
                result.extend(['H'] * (i - start))

        elif ss_list[i] == 'E':
            start = i
            while i < len(ss_list) and ss_list[i] == 'E':
                i += 1
            # Replace continuous E segments shorter than 2 with NaN.
            if i - start < 2:
                result.extend([np.nan] * (i - start))
            else:
                result.extend(['E'] * (i - start))

        elif ss_list[i] == 'C':
            start = i
            while i < len(ss_list) and ss_list[i] == 'C':
                i += 1
            # Replace continuous C segments shorter than 3 with NaN.
            if i - start < 3:
                result.extend([np.nan] * (i - start))
            else:
                result.extend(['C'] * (i - start))

        else:
            result.append(ss_list[i])
            i += 1

    return result

def align_dataframes(df1, df2):
    common_columns = list(set(df1.columns) & set(df2.columns))  # Find common columns.
    common_columns.sort()  # Sort common columns.

    # Store new columns in a dictionary.
    new_columns_dict = {}

    for column in common_columns:
        new_columns_dict[column + '_1'] = df1[column]  # Add the df1 column.
        new_columns_dict[column + '_2'] = df2[column]  # Add the df2 column.

    # Build the result DataFrame from the column dictionary.
    result_df = pd.DataFrame(new_columns_dict)


    return result_df

def calculate_means(grouped_df, mode="SS"):
    """
    Calculate mean conservation for secondary-structure categories or binary regions.

    Parameters
    ----------
    grouped_df : DataFrame
        Columns to be calculated.
    mode : str
        "SS" calculates means for H, E, and C.
        "Linker" or "Disorder" calculates means where the binary variable equals 1.

    Returns
    -------
    DataFrame
        Mean conservation values for the selected mode.
    """
    if mode == "SS":
        categories = ['H', 'E', 'C']
        result = {f"{cat}_Conservation": [] for cat in categories}

        for i in range(0, len(grouped_df.columns), 2):
            col1 = grouped_df.iloc[:, i].astype(str)  # SS column.
            col2 = grouped_df.iloc[:, i + 1]  # Numeric column.

            for cat in categories:
                result[f"{cat}_Conservation"].append(col2[col1 == cat].mean())

    elif mode in ["Linker", "Disorder"]:
        result = {f"{mode}_Conservation": []}

        for i in range(0, len(grouped_df.columns), 2):
            col1 = grouped_df.iloc[:, i].fillna(0).astype(int)  # Binary variable.
            col2 = grouped_df.iloc[:, i + 1]  # Numeric column.
            result[f"{mode}_Conservation"].append(col2[col1 == 1].mean())

    else:
        raise ValueError("mode must be 'SS', 'Linker', or 'Disorder'.")

    result = pd.DataFrame(result)
    result["AccessionName"] = [grouped_df.columns[i][:-2] for i in range(0, len(grouped_df.columns), 2)]
    result = result[["AccessionName"] + [col for col in result.columns if col != "AccessionName"]]
    result.reset_index(drop=True, inplace=True)

    return result

def Get_Conservation_features(hhm_file_path, linker_file_path, disorder_folder_path, SS_folder_path):
    # Calculate amino acid conservation as a DataFrame.
    relative_entropy_df = merge_relative_entropy_data(hhm_file_path)

    # Calculate sequence-level conservation.
    # relative_entropy_df_mean = relative_entropy_df.mean(axis=0)
    relative_entropy_df_mean = relative_entropy_df.mean(axis=0, skipna=True)
    relative_entropy_df_mean = relative_entropy_df_mean.reset_index()
    relative_entropy_df_mean.columns = ["AccessionName", "sequence_Conservation"]

    # Calculate linker conservation.
    linker_processed_data = process_single_apod_file(linker_file_path)
    linker_processed_df = processed_data_to_dataframe(linker_processed_data)
    Linker_aligned_df = align_dataframes(linker_processed_df, relative_entropy_df)
    linker_summary_df = calculate_means(Linker_aligned_df, mode="Linker")

    # Calculate disorder conservation.
    disorder_processed_df = process_folder(disorder_folder_path)
    Disorder_aligned_df = align_dataframes(disorder_processed_df, relative_entropy_df)
    disorder_summary_df = calculate_means(Disorder_aligned_df, mode="Disorder")

    # Calculate secondary-structure conservation.
    df = extract_second_column_to_dataframe(SS_folder_path)
    df = df.iloc[1:].reset_index(drop=True)
    SS_processed_df = df.apply(lambda col: process_ss_column(col.tolist()), axis=0)
    SS_aligned_df = align_dataframes(SS_processed_df, relative_entropy_df)
    SS_summary_df = calculate_means(SS_aligned_df, mode="SS")

    # Merge features.
    result = relative_entropy_df_mean.copy()
    result = result.merge(linker_summary_df, on="AccessionName")
    result = result.merge(disorder_summary_df, on="AccessionName")
    result = result.merge(SS_summary_df, on="AccessionName")
    
    result.rename(columns={
        "sequence_Conservation": "CONS_SEQ",
        "Linker_Conservation": "CONS_FL",
        "Disorder_Conservation": "CONS_IDR",
        "H_Conservation": "CONS_α",
        "E_Conservation": "CONS_β",
        "C_Conservation": "CONS_γ",
    }, inplace=True)


    return result

def Merge_Features(positive_input_folder, positive_apod_file_path, positive_disorder_folder_path, positive_root_folder, negative_input_folder, negative_apod_file_path, negative_disorder_folder_path, negative_root_folder):
    positive_features_merged_df = Get_Conservation_features(positive_input_folder, positive_apod_file_path, positive_disorder_folder_path, positive_root_folder)
    negative_features_merged_df = Get_Conservation_features(negative_input_folder, negative_apod_file_path, negative_disorder_folder_path, negative_root_folder)
    positive_features_merged_df["label"] = 1
    negative_features_merged_df["label"] = 0
    # print("positive_features_merged_df", positive_features_merged_df)
    # print("negative_features_merged_df", negative_features_merged_df)

    combined_df = pd.concat([positive_features_merged_df, negative_features_merged_df], axis=0).reset_index(drop=True)
    combined_df = combined_df.sort_values(by=["label", "AccessionName"], ascending=[False, True]).reset_index(drop=True)
    combined_df.fillna(0, inplace=True)
    # combined_df.index.name = "AccessionName"
    return combined_df


def get_manifest_conservation(manifest_path, linker_path, records, require_complete=True):
    """Preserve the original calculations while indexing inputs through one manifest."""
    background = {
        "A": 0.0834, "C": 0.0157, "D": 0.0541, "E": 0.061, "F": 0.0384,
        "G": 0.0668, "H": 0.0228, "I": 0.0533, "K": 0.052, "L": 0.0925,
        "M": 0.0218, "N": 0.0428, "P": 0.0523, "Q": 0.0401, "R": 0.0598,
        "S": 0.079, "T": 0.0583, "V": 0.0631, "W": 0.0127, "Y": 0.0287,
    }
    linker_by_accession = dict(process_single_apod_file(linker_path))
    manifest = pd.read_csv(manifest_path)
    results = []
    required = manifest[["hhm_available", "disorder_available", "ss_available", "linker_available"]].all(axis=1)
    for _, row in manifest.loc[required].iterrows():
        hhm_path = resolve_project_path(PROJECT_ROOT, row.hhm_raw_path)
        disorder_path = resolve_project_path(PROJECT_ROOT, row.disorder_raw_path)
        ss_path = resolve_project_path(PROJECT_ROOT, row.ss_raw_path)

        entropy = calculate_relative_entropy_from_hhm(hhm_path, background)["relative_entropy"].reset_index(drop=True)
        linker = pd.Series(list(linker_by_accession[row.accession])).fillna(0).astype(int)
        disorder = pd.Series(classify_amino_acids(disorder_path)).fillna(0).astype(int)
        ss_frame = pd.read_csv(ss_path, sep=r"\s+", header=None, on_bad_lines="skip")
        ss_values = pd.Series(process_ss_column(ss_frame.iloc[:, 2].iloc[1:].reset_index(drop=True).tolist()))

        linker_entropy = pd.concat([linker, entropy], axis=1)
        disorder_entropy = pd.concat([disorder, entropy], axis=1)
        ss_entropy = pd.concat([ss_values, entropy], axis=1)
        results.append({
            "AccessionName": row.accession,
            "CONS_SEQ": entropy.mean(skipna=True),
            "CONS_FL": linker_entropy.loc[linker_entropy.iloc[:, 0] == 1].iloc[:, 1].mean(),
            "CONS_IDR": disorder_entropy.loc[disorder_entropy.iloc[:, 0] == 1].iloc[:, 1].mean(),
            "CONS_α": ss_entropy.loc[ss_entropy.iloc[:, 0].astype(str) == "H"].iloc[:, 1].mean(),
            "CONS_β": ss_entropy.loc[ss_entropy.iloc[:, 0].astype(str) == "E"].iloc[:, 1].mean(),
            "CONS_γ": ss_entropy.loc[ss_entropy.iloc[:, 0].astype(str) == "C"].iloc[:, 1].mean(),
        })
    frame = pd.DataFrame(results)
    return finalize_features(frame, records, require_complete=require_complete)

if __name__ == "__main__":
    config = dataset_config(args.dataset)
    records = read_optional_label_fasta(config["fasta"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    output = output_name(config, "Conservation")
    result = get_manifest_conservation(config["manifest"], config["linker"], records, not args.allow_missing)
    result.to_csv(output, index=False, encoding="utf-8-sig")
#    print(f"Wrote {len(result)} rows to {output}")
