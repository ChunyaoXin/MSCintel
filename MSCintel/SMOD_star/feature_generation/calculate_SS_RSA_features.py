# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from Parameter import args, dataset_config, PROJECT_ROOT
from workflow_utils import read_optional_label_fasta, finalize_features, resolve_project_path
from sequence_feature_common import output_name
class SSRSAFeatureExtractor:
    def __init__(self, positive_ss_predicted_file, negative_ss_predicted_file):
        # 初始化类的一些参数
        self.max_asa_values = {
            'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148, 'Q': 214, 'E': 214,
            'G': 97, 'H': 216, 'I': 195, 'L': 191, 'K': 230, 'M': 203, 'F': 228,
            'P': 154, 'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165
        }

        # 文件路径作为类的属性
        self.positive_ss_predicted_file = positive_ss_predicted_file
        self.negative_ss_predicted_file = negative_ss_predicted_file

    def process_ss_column(self, ss_list):
        """
        处理 SS 列，将长度小于 3 的连续 H、E 和 C 片段替换为 NaN
        保留 NaN 不进行处理
        """
        result = []
        i = 0
        while i < len(ss_list):
            # 跳过 NaN 值
            if pd.isna(ss_list[i]):
                result.append(ss_list[i])
                i += 1
                continue

            if ss_list[i] == 'H':
                start = i
                while i < len(ss_list) and ss_list[i] == 'H':
                    i += 1
                # 如果连续 H 的长度小于 3，替换为 NaN
                if i - start < 3:
                    result.extend([np.nan] * (i - start))
                else:
                    result.extend(['H'] * (i - start))

            elif ss_list[i] == 'E':
                start = i
                while i < len(ss_list) and ss_list[i] == 'E':
                    i += 1
                # 如果连续 E 的长度小于 2，替换为 NaN
                if i - start < 2:
                    result.extend([np.nan] * (i - start))
                else:
                    result.extend(['E'] * (i - start))

            elif ss_list[i] == 'C':
                start = i
                while i < len(ss_list) and ss_list[i] == 'C':
                    i += 1
                # 如果连续 C 的长度小于 3，替换为 NaN
                if i - start < 3:
                    result.extend([np.nan] * (i - start))
                else:
                    result.extend(['C'] * (i - start))

            else:
                result.append(ss_list[i])
                i += 1

        return result

    def convert_asa_to_rsa(self, aa_series, asa_series):
        """
        将 ASA 转换为 RSA
        :param aa_series: 氨基酸序列 (pd.Series)
        :param asa_series: ASA 列 (pd.Series)
        :return: RSA 列 (pd.Series)
        """
        rsa_values = []
        for aa, asa in zip(aa_series, asa_series):
            max_asa = self.max_asa_values.get(aa, 1)  # 获取对应氨基酸的 MaxASA，默认值为 1 避免除零
            rsa_values.append(asa / max_asa if max_asa > 0 else 0)
        return pd.Series(rsa_values)

    def calculate_sequence_rsa_mean(self, rsa_series):
        """
        计算序列的 RSA 平均值
        :param rsa_series: RSA 列 (pd.Series)
        :return: 平均 RSA (float)
        """
        return rsa_series.mean()

    def calculate_avg_rsa(self, rsa_list, ss_list):
        """
        计算 ss_list 中 H、E、C 三类的 RSA 平均值
        """
        h_rsa = [rsa_list[i] for i in range(len(ss_list)) if ss_list[i] == 'H']
        e_rsa = [rsa_list[i] for i in range(len(ss_list)) if ss_list[i] == 'E']
        c_rsa = [rsa_list[i] for i in range(len(ss_list)) if ss_list[i] == 'C']

        # 计算均值，如果列表为空则返回 NaN
        h_avg = np.mean(h_rsa) if h_rsa else np.nan
        e_avg = np.mean(e_rsa) if e_rsa else np.nan
        c_avg = np.mean(c_rsa) if c_rsa else np.nan

        return h_avg, e_avg, c_avg

    def extract_columns(self, root_folder):
        """
        遍历 root_folder 下的每个子文件夹，提取 seq.phipsi 文件的 1, 2, 10 列，
        计算 H、E、C 片段的 RSA 平均值，并返回一个 DataFrame
        """
        results = []
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file == "seq.phipsi":
                    file_path = os.path.join(subdir, file)
                    subfolder_name = os.path.basename(subdir)
                    try:
                        # 读取文件并提取所需列
                        df = pd.read_csv(file_path, delim_whitespace=True)
                        extracted_df = df.iloc[:, [1, 2, 10]].copy()
                        extracted_df.columns = ["AA", "SS", "ASA"]
                        extracted_df["SS_changed"] = self.process_ss_column(extracted_df["SS"].tolist())
                        extracted_df["RSA"] = self.convert_asa_to_rsa(extracted_df["AA"], extracted_df["ASA"])

                        # 计算 H、E、C 的 RSA 平均值
                        h_avg_rsa, e_avg_rsa, c_avg_rsa = self.calculate_avg_rsa(
                            extracted_df["RSA"].tolist(), extracted_df["SS_changed"]
                        )

                        # 计算全局 RSA 均值
                        rsa_mean = np.nanmean(extracted_df["RSA"])

                        # 存入结果
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
