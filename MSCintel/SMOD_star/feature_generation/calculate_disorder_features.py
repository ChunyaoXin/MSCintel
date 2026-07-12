import os
import pandas as pd
import numpy as np
from Parameter import args, dataset_config, PROJECT_ROOT
from workflow_utils import read_optional_label_fasta, finalize_features, resolve_project_path
from sequence_feature_common import output_name


class DisorderFeatureExtractor:
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path

    def classify_amino_acids(self, file_path):
        """
        读取并分类氨基酸序列
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 过滤掉注释和标题行
        lines = [line.strip() for line in lines if not line.startswith('#') and not line.startswith('Num')]
        sequence = ''
        classifications = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            score = float(parts[2])
            classification = 'D' if score > 0.576 else 'O'
            classifications.append((parts[0], parts[1], parts[2], classification))
            sequence += classification

        # 处理连续 D 片段长度小于 4 的情况
        processed_sequence = list(sequence)
        i = 0
        while i < len(processed_sequence):
            if processed_sequence[i] == 'D':
                start = i
                while i < len(processed_sequence) and processed_sequence[i] == 'D':
                    i += 1
                end = i
                if end - start < 4:  # 长度小于 4 的片段转换为 'O'
                    for j in range(start, end):
                        processed_sequence[j] = 'O'
            else:
                i += 1

        sequence = ''.join(processed_sequence)
        total_sequence_length = len(sequence)
        first_d_position = sequence.find('D') + 1
        last_d_position = sequence.rfind('D') + 1

        first_d_position_ratio = first_d_position / total_sequence_length if total_sequence_length > 0 else 0
        last_d_position_ratio = last_d_position / total_sequence_length if total_sequence_length > 0 else 0

        proportions = {'D': sequence.count('D') / total_sequence_length if total_sequence_length > 0 else 0}

        return sequence, total_sequence_length, proportions, first_d_position_ratio, last_d_position_ratio

    def calculate_fragment_statistics(self, sequence, total_sequence_length, char, accession_name):
        """
        计算最长、最短、平均片段的比例以及最长片段的位置
        """
        fragments = []
        start_positions = []
        end_positions = []
        current_fragment = 0
        start_index = None

        for i, c in enumerate(sequence):
            if c == char:
                if current_fragment == 0:
                    start_index = i
                current_fragment += 1
            else:
                if current_fragment > 0:
                    fragments.append(current_fragment)
                    start_positions.append(start_index)
                    end_positions.append(i - 1)
                current_fragment = 0

        if current_fragment > 0:
            fragments.append(current_fragment)
            start_positions.append(start_index)
            end_positions.append(len(sequence) - 1)

        #计算总片段数（基于替换后的序列）
        total_segments = 0
        prev_char = None
        for c in sequence:
            if c != prev_char:  # 字符变化时计数
                total_segments += 1
                prev_char = c

        if not fragments:
            return 0, 0, 0, 0, 0, 0, accession_name

        max_length = max(fragments)
        max_indices = [i for i, length in enumerate(fragments) if length == max_length]
        max_start_ratios = [(start_positions[i] + 1) / total_sequence_length for i in max_indices]
        max_end_ratios = [(end_positions[i] + 1) / total_sequence_length for i in max_indices]

        max_start_ratio = max_start_ratios[0]
        max_end_ratio = max_end_ratios[0]

        max_proportion = max_length / total_sequence_length if total_sequence_length > 0 else 0
        min_proportion = min(fragments) / total_sequence_length if total_sequence_length > 0 else 0
        avg_proportion = sum(fragments) / len(fragments) / total_sequence_length if total_sequence_length > 0 else 0

        # 计算 D 片段占所有片段的比例
        num_d_segments = len(fragments)
        proportion_d_segments = num_d_segments / total_segments if total_segments > 0 else 0

        return max_proportion, min_proportion, avg_proportion, max_start_ratio, max_end_ratio, proportion_d_segments, accession_name


    def process_files_in_folder(self, folder_path):
        """
        处理文件夹中的所有 txt 文件，返回数据框
        """
        result_data = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                sequence, total_sequence_length, proportions, first_d_position_ratio, last_d_position_ratio = self.classify_amino_acids(
                    file_path)

                #accession_name = filename.removesuffix('.txt')
                accession_name = filename.replace('.txt', '') if filename.endswith('.txt') else filename

                max_d_proportion, min_d_proportion, avg_d_proportion, max_d_start_ratio, max_d_end_ratio, proportion_d_segments, accession_name = self.calculate_fragment_statistics(
                    sequence, total_sequence_length, char="D", accession_name=accession_name)

                result_data.append([
                    accession_name, proportions['D'], proportion_d_segments, max_d_proportion, min_d_proportion, avg_d_proportion,
                    first_d_position_ratio, last_d_position_ratio, max_d_start_ratio, max_d_end_ratio
                ])

        columns = ["AccessionName", "IDR_AA%", "IDR%", "IDR_MAXlen", "IDR_MINlen", "IDR_AVGlen", "IDR_S", "IDR_E", "MAX_IDR_S", "MAX_IDR_E"]
        return pd.DataFrame(result_data, columns=columns)

    def get_disorder_features(self, records, require_complete=True):
        manifest = pd.read_csv(self.manifest_path)
        result_data = []
        for _, row in manifest.loc[manifest["disorder_available"]].iterrows():
            path = resolve_project_path(PROJECT_ROOT, row["disorder_raw_path"])
            sequence, total_length, proportions, first_position, last_position = self.classify_amino_acids(path)
            longest, shortest, average, longest_start, longest_end, fragment_ratio, _ = self.calculate_fragment_statistics(
                sequence, total_length, char="D", accession_name=row.accession
            )
            result_data.append([
                row.accession, proportions["D"], fragment_ratio, longest, shortest, average,
                first_position, last_position, longest_start, longest_end,
            ])
        columns = ["AccessionName", "IDR_AA%", "IDR%", "IDR_MAXlen", "IDR_MINlen", "IDR_AVGlen", "IDR_S", "IDR_E", "MAX_IDR_S", "MAX_IDR_E"]
        frame = pd.DataFrame(result_data, columns=columns)
        return finalize_features(frame, records, require_complete=require_complete)


if __name__ == "__main__":
    config = dataset_config(args.dataset)
    records = read_optional_label_fasta(config["fasta"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    output = output_name(config, "Disorder")
    extractor = DisorderFeatureExtractor(config["manifest"])
    frame = extractor.get_disorder_features(records, not args.allow_missing)
    frame.to_csv(output, index=False)
#    print(f"Wrote {len(frame)} rows to {output}")
