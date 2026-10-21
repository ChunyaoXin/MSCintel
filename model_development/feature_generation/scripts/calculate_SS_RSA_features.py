# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from Parameter import args, dataset_config, PROJECT_ROOT
from workflow_utils import read_optional_label_fasta, finalize_features, resolve_project_path
from sequence_feature_common import output_name
class SSRSAFeatureExtractor:
    def __init__(self, positive_ss_predicted_file, negative_ss_predicted_file):
        # Initialize class parameters
        self.max_asa_values = {
            'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148, 'Q': 214, 'E': 214,
            'G': 97, 'H': 216, 'I': 195, 'L': 191, 'K': 230, 'M': 203, 'F': 228,
            'P': 154, 'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165
        }

        # Store file paths as class attributes
        self.positive_ss_predicted_file = positive_ss_predicted_file
        self.negative_ss_predicted_file = negative_ss_predicted_file

    def process_ss_column(self, ss_list):
        """
        Process the SS column by replacing consecutive H and C segments shorter than 3
        and consecutive E segments shorter than 2 with NaN.
        Preserve existing NaN values without modification.
        """
        result = []
        i = 0
        while i < len(ss_list):
            # Skip NaN values
            if pd.isna(ss_list[i]):
                result.append(ss_list[i])
                i += 1
                continue

            if ss_list[i] == 'H':
                start = i
                while i < len(ss_list) and ss_list[i] == 'H':
                    i += 1
                # Replace H segments shorter than 3 with NaN
                if i - start < 3:
                    result.extend([np.nan] * (i - start))
                else:
                    result.extend(['H'] * (i - start))

            elif ss_list[i] == 'E':
                start = i
                while i < len(ss_list) and ss_list[i] == 'E':
                    i += 1
                # Replace E segments shorter than 2 with NaN
                if i - start < 2:
                    result.extend([np.nan] * (i - start))
                else:
                    result.extend(['E'] * (i - start))

            elif ss_list[i] == 'C':
                start = i
                while i < len(ss_list) and ss_list[i] == 'C':
                    i += 1
                # Replace C segments shorter than 3 with NaN
                if i - start < 3:
                    result.extend([np.nan] * (i - start))
                else:
                    result.extend(['C'] * (i - start))

            else:
                result.append(ss_list[i])
                i += 1

        return result

    def convert_asa_to_rsa(self, aa_series, asa_series):
        rsa_values = []
        for aa, asa in zip(aa_series, asa_series):
            max_asa = self.max_asa_values.get(aa, 1)  # get aa MaxASA, default is 1 to aviod /0
            rsa_values.append(asa / max_asa if max_asa > 0 else 0)
        return pd.Series(rsa_values)

    def calculate_sequence_rsa_mean(self, rsa_series):
        return rsa_series.mean()

    def calculate_avg_rsa(self, rsa_list, ss_list):
        """
        calculate mean RSA of H, E, C in ss_list
        """
        h_rsa = [rsa_list[i] for i in range(len(ss_list)) if ss_list[i] == 'H']
        e_rsa = [rsa_list[i] for i in range(len(ss_list)) if ss_list[i] == 'E']
        c_rsa = [rsa_list[i] for i in range(len(ss_list)) if ss_list[i] == 'C']

        # Calculate the mean; return NaN if the list is empty
        h_avg = np.mean(h_rsa) if h_rsa else np.nan
        e_avg = np.mean(e_rsa) if e_rsa else np.nan
        c_avg = np.mean(c_rsa) if c_rsa else np.nan

        return h_avg, e_avg, c_avg

    def extract_columns(self, root_folder):
        results = []
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file == "seq.phipsi":
                    file_path = os.path.join(subdir, file)
                    subfolder_name = os.path.basename(subdir)
                    try:
                        # read file and extract required column
                        df = pd.read_csv(file_path, delim_whitespace=True)
                        extracted_df = df.iloc[:, [1, 2, 10]].copy()
                        extracted_df.columns = ["AA", "SS", "ASA"]
                        extracted_df["SS_changed"] = self.process_ss_column(extracted_df["SS"].tolist())
                        extracted_df["RSA"] = self.convert_asa_to_rsa(extracted_df["AA"], extracted_df["ASA"])

                        # calculate mean RSA of H, E, C
                        h_avg_rsa, e_avg_rsa, c_avg_rsa = self.calculate_avg_rsa(
                            extracted_df["RSA"].tolist(), extracted_df["SS_changed"]
                        )

                        # calculate global mean RSA
                        rsa_mean = np.nanmean(extracted_df["RSA"])

                        # save result
                        results.append({
                            "AccessionName": subfolder_name,
                            "AVG_RSA_α": h_avg_rsa,
                            "AVG_RSA_β": e_avg_rsa,
                            "AVG_RSA_γ": c_avg_rsa,
                            "AVG_RSA_SEQ": rsa_mean,
                        })
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        results = pd.DataFrame(results)
        return results

    def get_ss_rsa_features(self):
        positive_ss_rsa_df = self.extract_columns(self.positive_ss_predicted_file)
        negative_ss_rsa_df = self.extract_columns(self.negative_ss_predicted_file)
        positive_ss_rsa_df['label'] = 1
        negative_ss_rsa_df['label'] = 0
        combined_df = pd.concat([positive_ss_rsa_df, negative_ss_rsa_df], axis=0, ignore_index=True).fillna(0)
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
                extracted = df.iloc[:, [1, 2, 10]].copy()
                extracted.columns = ["AA", "SS", "ASA"]
                changed = self.process_ss_column(extracted["SS"].tolist())
                rsa = self.convert_asa_to_rsa(extracted["AA"], extracted["ASA"])
                h_avg, e_avg, c_avg = self.calculate_avg_rsa(rsa.tolist(), changed)
                results.append({
                    "AccessionName": row.accession,
                    "AVG_RSA_α": h_avg,
                    "AVG_RSA_β": e_avg,
                    "AVG_RSA_γ": c_avg,
                    "AVG_RSA_SEQ": np.nanmean(rsa),
                })
            except Exception as exc:
                raise ValueError(f"Error processing {file_path}: {exc}") from exc
        frame = pd.DataFrame(results)
        return finalize_features(frame, records, require_complete=require_complete)

if __name__ == "__main__":
    config = dataset_config(args.dataset)
    records = read_optional_label_fasta(config["fasta"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    output = output_name(config, "SS_RSA")
    extractor = SSRSAFeatureExtractor(None, None)
    frame = extractor.get_manifest_features(config["manifest"], records, not args.allow_missing)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
#    print(f"Wrote {len(frame)} rows to {output}")
