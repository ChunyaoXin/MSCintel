# -*- coding: utf-8 -*-

import os
import pandas as pd
from Parameter import args, dataset_config, PROJECT_ROOT
from workflow_utils import read_optional_label_fasta, finalize_features, resolve_project_path
from sequence_feature_common import output_name

class SSFeatureExtractor:
    def __init__(self, positive_root_folder, negative_root_folder):
        self.positive_root_folder = positive_root_folder
        self.negative_root_folder = negative_root_folder

    def process_ss_column(self, ss_list):
        """
        Process the SS column by replacing consecutive H segments shorter than 3 and E segments shorter than 2 with C
        """
        result = []
        i = 0
        while i < len(ss_list):
            if ss_list[i] == 'H':
                start = i
                while i < len(ss_list) and ss_list[i] == 'H':
                    i += 1
                if i - start < 3:
                    result.extend(['C'] * (i - start))
                else:
                    result.extend(['H'] * (i - start))
            elif ss_list[i] == 'E':
                start = i
                while i < len(ss_list) and ss_list[i] == 'E':
                    i += 1
                if i - start < 2:
                    result.extend(['C'] * (i - start))
                else:
                    result.extend(['E'] * (i - start))
            else:
                result.append(ss_list[i])
                i += 1
        return result

    def calculate_h_e_features_with_c(self, ss_changed):
        """
        Calculate H, E, and C segment features from the SS_changed column.
        Include segment counts, lengths, related ratios, and position-based features.
        """
        total_length = len(ss_changed)

        # Ratios of H, E, and C
        h_count = ss_changed.count('H')
        e_count = ss_changed.count('E')
        h_ratio = h_count / total_length if total_length > 0 else 0
        e_ratio = e_count / total_length if total_length > 0 else 0

        # Split by segment type
        fragments = []
        current_fragment = []
        fragment_start_indices = []

        for i, char in enumerate(ss_changed):
            if current_fragment and char != current_fragment[-1]:  # Start of a new segment
                fragments.append((current_fragment[0], len(current_fragment)))
                fragment_start_indices.append(i - len(current_fragment))
                current_fragment = [char]
            else:
                current_fragment.append(char)
        # Add the last segment
        if current_fragment:
            fragments.append((current_fragment[0], len(current_fragment)))
            fragment_start_indices.append(total_length - len(current_fragment))

        # Group by type
        h_fragments = [(start, length) for (char, length), start in zip(fragments, fragment_start_indices) if char == 'H']
        e_fragments = [(start, length) for (char, length), start in zip(fragments, fragment_start_indices) if char == 'E']
        c_fragments = [(start, length) for (char, length), start in zip(fragments, fragment_start_indices) if char == 'C']

        # Extract segment lengths and counts
        h_fragment_lengths = [length for _, length in h_fragments]
        e_fragment_lengths = [length for _, length in e_fragments]

        h_fragment_count = len(h_fragments)
        e_fragment_count = len(e_fragments)
        c_fragment_count = len(c_fragments)

        total_fragment_count = h_fragment_count + e_fragment_count + c_fragment_count

        # Ratios of each segment type
        h_fragment_ratio = h_fragment_count / total_fragment_count if total_fragment_count > 0 else 0
        e_fragment_ratio = e_fragment_count / total_fragment_count if total_fragment_count > 0 else 0

        # H segment features
        shortest_h_fragment = min(h_fragment_lengths, default=0)
        longest_h_fragment = max(h_fragment_lengths, default=0)
        average_h_fragment = sum(h_fragment_lengths) / h_fragment_count if h_fragment_count > 0 else 0
        longest_h_fragment_index = h_fragments[h_fragment_lengths.index(longest_h_fragment)] if longest_h_fragment > 0 else None

        # E segment features
        shortest_e_fragment = min(e_fragment_lengths, default=0)
        longest_e_fragment = max(e_fragment_lengths, default=0)
        average_e_fragment = sum(e_fragment_lengths) / e_fragment_count if e_fragment_count > 0 else 0
        longest_e_fragment_index = e_fragments[e_fragment_lengths.index(longest_e_fragment)] if longest_e_fragment > 0 else None

        # Normalize H, E, and C segment features
        shortest_h_fragment_ratio = shortest_h_fragment / total_length if total_length > 0 else 0
        longest_h_fragment_ratio = longest_h_fragment / total_length if total_length > 0 else 0
        average_h_fragment_ratio = average_h_fragment / total_length if total_length > 0 else 0

        shortest_e_fragment_ratio = shortest_e_fragment / total_length if total_length > 0 else 0
        longest_e_fragment_ratio = longest_e_fragment / total_length if total_length > 0 else 0
        average_e_fragment_ratio = average_e_fragment / total_length if total_length > 0 else 0

        # Positions of the first and last H
        first_h_position = (ss_changed.index('H') + 1) / total_length if 'H' in ss_changed else 0
        last_h_position = (len(ss_changed) - ss_changed[::-1].index('H')) / total_length if 'H' in ss_changed else 0

        # Positions of the first and last E
        first_e_position = (ss_changed.index('E') + 1) / total_length if 'E' in ss_changed else 0
        last_e_position = (len(ss_changed) - ss_changed[::-1].index('E')) / total_length if 'E' in ss_changed else 0

        # Start and end positions of the longest H and E segments
        longest_h_start = (longest_h_fragment_index[0] + 1) / total_length if longest_h_fragment_index else 0
        longest_h_end = (longest_h_fragment_index[0] + longest_h_fragment_index[1]) / total_length if longest_h_fragment_index else 0
        longest_e_start = (longest_e_fragment_index[0] + 1) / total_length if longest_e_fragment_index else 0
        longest_e_end = (longest_e_fragment_index[0] + longest_e_fragment_index[1]) / total_length if longest_e_fragment_index else 0

        return {
            "α_AA%": h_ratio,
            "β_AA%": e_ratio,
            "α%": h_fragment_ratio,
            "β%": e_fragment_ratio,
            "α_MINlen": shortest_h_fragment_ratio,
            "α_MAXlen": longest_h_fragment_ratio,
            "α_AVGlen": average_h_fragment_ratio,
            "β_MINlen": shortest_e_fragment_ratio,
            "β_MAXlen": longest_e_fragment_ratio,
            "β_AVGlen": average_e_fragment_ratio,
            "α_S": first_h_position,
            "α_E": last_h_position,
            "β_S": first_e_position,
            "β_E": last_e_position,
            "MAX_α_S": longest_h_start,
            "MAX_α_E": longest_h_end,
            "MAX_β_S": longest_e_start,
            "MAX_β_E": longest_e_end,
        }

    def extract_columns(self, root_folder):
        """
        Traverse each subfolder under root_folder and extract columns 1, 2, and 10 from seq.phipsi,
        rename them to "AA", "SS", and "ASA", process the SS column, calculate RSA, and save the results as CSV files named after each subfolder
        """
        results = []
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file == "seq.phipsi":
                    file_path = os.path.join(subdir, file)
                    subfolder_name = os.path.basename(subdir)
                    try:
                        # Read the file, extract the required columns, and rename them
                        df = pd.read_csv(file_path, delim_whitespace=True)
                        extracted_df = df.iloc[:, [1, 2, 10]].copy()
                        # Rename columns
                        extracted_df.columns = ["AA", "SS", "ASA"]
                        # Process the SS column and generate a new column
                        ss_changed = self.process_ss_column(extracted_df["SS"].tolist())
                        extracted_df["SS_changed"] = ss_changed
                        # Calculate H and E segment features
                        features = self.calculate_h_e_features_with_c(ss_changed)
                        # Integrate features into the DataFrame and add the AccessionName column
                        feature_row = {'AccessionName': subfolder_name}  # Add the sequence name
                        feature_row.update(features)  # Add features to the row
                        results.append(feature_row)

                        # Integrate features into the DataFrame
                        for feature_name, feature_value in features.items():
                            extracted_df[feature_name] = feature_value

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        # Save features from all sequences to a single CSV file
        all_results_df = pd.DataFrame(results)
        return all_results_df

    def get_ss_features(self):
        positive_features_df = self.extract_columns(self.positive_root_folder)
        negative_features_df = self.extract_columns(self.negative_root_folder)
        positive_features_df['label'] = 1
        negative_features_df['label'] = 0
        # Concatenate the two DataFrames row-wise
        combined_df = pd.concat([positive_features_df, negative_features_df], axis=0, ignore_index=True)
        # The ignore_index=True parameter resets the index of the combined DataFrame to a continuous index instead of retaining the original indices.
        #combined_df.set_index(combined_df.columns[0], inplace=True)
        combined_df = combined_df.sort_values(by=['label', 'AccessionName'], ascending=[False, True])
        combined_df.set_index('AccessionName', inplace=True)
        combined_df.fillna(0, inplace=True)


        return combined_df

    def get_manifest_features(self, manifest_path, records, require_complete=True):
        manifest = pd.read_csv(manifest_path)
        results = []
        for _, row in manifest.loc[manifest["ss_available"]].iterrows():
            file_path = resolve_project_path(PROJECT_ROOT, row["ss_raw_path"])
            try:
                df = pd.read_csv(file_path, sep=r"\s+")
                extracted_df = df.iloc[:, [1, 2, 10]].copy()
                extracted_df.columns = ["AA", "SS", "ASA"]
                features = self.calculate_h_e_features_with_c(
                    self.process_ss_column(extracted_df["SS"].tolist())
                )
                results.append({"AccessionName": row.accession, **features})
            except Exception as exc:
                raise ValueError(f"Error processing {file_path}: {exc}") from exc
        frame = pd.DataFrame(results)
        return finalize_features(frame, records, require_complete=require_complete)

if __name__ == "__main__":
    config = dataset_config(args.dataset)
    records = read_optional_label_fasta(config["fasta"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    output = output_name(config, "SS")
    extractor = SSFeatureExtractor(None, None)
    frame = extractor.get_manifest_features(config["manifest"], records, not args.allow_missing)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
#    print(f"Wrote {len(frame)} rows to {output}")
