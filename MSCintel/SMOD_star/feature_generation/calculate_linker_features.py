#####下面是写为类的形式

# -*- coding: utf-8 -*-
import pandas as pd
import os
from Parameter import args, dataset_config
from workflow_utils import read_optional_label_fasta, finalize_features
from sequence_feature_common import output_name

class LinkerFeatureExtractor:
    def __init__(self, apod_file):
        self.apod_file = apod_file

    def replace_short_segments(self, sequence):
        new_sequence = ''
        current_length = 0

        for char in sequence:
            if char == '1':
                current_length += 1
            else:
                if current_length > 0:
                    if current_length < 4:
                        new_sequence += '0' * current_length
                    else:
                        new_sequence += '1' * current_length
                    current_length = 0
                new_sequence += char

        if current_length > 0:
            if current_length < 4:
                new_sequence += '0' * current_length
            else:
                new_sequence += '1' * current_length

        return new_sequence

    def extract_and_process_apod(self, apod_file_path):
        with open(apod_file_path, 'r') as file:
            lines = file.readlines()

        processed_data = []
        current_accession = None
        current_lines = []

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('>'):
                # 处理前一个组
                if current_accession is not None:
                    # 确保至少有后续两行数据（原i+1和i+2行）
                    if len(current_lines) >= 2:
                        binary_value = current_lines[1]  # 原i+2对应这里的第二个元素
                        new_sequence = self.replace_short_segments(binary_value)
                        processed_data.append({
                            'AccessionName': current_accession,
                            'Sequence': new_sequence
                        })
                # 新组开始
                current_accession = stripped_line.lstrip('>').strip()
                current_lines = []
            else:
                if current_accession is not None:  # 确保在有效组内
                    current_lines.append(stripped_line)

        # 处理文件最后一个组
        if current_accession is not None and len(current_lines) >= 2:
            binary_value = current_lines[1]
            new_sequence = self.replace_short_segments(binary_value)
            processed_data.append({
                'AccessionName': current_accession,
                'Sequence': new_sequence
            })

        return processed_data
    def calculate_linker_features(self, sequence):
        total_length = len(sequence)
        one_count = sequence.count('1')
        one_ratio = one_count / total_length if total_length > 0 else 0

        fragments = sequence.split('0')
        one_fragments = [len(frag) for frag in fragments if frag]
        zero_fragments = [len(frag) for frag in sequence.split('1') if frag]

        one_fragment_count = len(one_fragments)
        zero_fragment_count = len(zero_fragments)
        one_fragment_ratio = one_fragment_count / (one_fragment_count + zero_fragment_count) if (
                                                                                                        one_fragment_count + zero_fragment_count) > 0 else 0

        shortest_one_fragment = min(one_fragments, default=0)
        longest_one_fragment = max(one_fragments, default=0)
        average_one_fragment = sum(one_fragments) / one_fragment_count if one_fragment_count > 0 else 0

        shortest_one_fragment_ratio = shortest_one_fragment / total_length if total_length > 0 else 0
        longest_one_fragment_ratio = longest_one_fragment / total_length if total_length > 0 else 0
        average_one_fragment_ratio = average_one_fragment / total_length if total_length > 0 else 0

        first_one_position = (sequence.find('1') + 1) / total_length if total_length > 0 else 0
        last_one_index = [index for index, char in reversed(list(enumerate(sequence))) if char == '1']
        last_one_index = last_one_index[0] if last_one_index else -1
        last_one_position = (last_one_index + 1) / total_length if total_length > 0 else 0

        if longest_one_fragment > 0:
            longest_one_fragment_start = None
            longest_one_fragment_end = None
            current_start = None
            current_length = 0
            max_length = 0

            for i, char in enumerate(sequence):
                if char == '1':
                    if current_start is None:
                        current_start = i
                    current_length += 1
                else:
                    if current_length > max_length:
                        max_length = current_length
                        longest_one_fragment_start = current_start
                        longest_one_fragment_end = i - 1
                    current_length = 0
                    current_start = None

            if current_length > max_length:
                longest_one_fragment_start = current_start
                longest_one_fragment_end = len(sequence) - 1

            if longest_one_fragment_start is not None and longest_one_fragment_end is not None:
                first_in_longest = (longest_one_fragment_start + 1) / total_length if total_length > 0 else 0
                last_in_longest = (longest_one_fragment_end + 1) / total_length if total_length > 0 else 0
            else:
                first_in_longest = 0
                last_in_longest = 0
        else:
            first_in_longest = 0
            last_in_longest = 0

        return {
            "FL_AA%": one_ratio,
            "FL%": one_fragment_ratio,
            "FL_MINlen": shortest_one_fragment_ratio,
            "FL_MAXlen": longest_one_fragment_ratio,
            "FL_AVGlen": average_one_fragment_ratio,
            "FL_S": first_one_position,
            "FL_E": last_one_position,
            "MAX_FL_S": first_in_longest,
            "MAX_FL_E": last_in_longest,
        }

    def process_single_sequence_file(self, apod_file_path):
        processed_data = self.extract_and_process_apod(apod_file_path)

        results = []
        for entry in processed_data:
            features = self.calculate_linker_features(entry['Sequence'])
            features['AccessionName'] = entry['AccessionName']
            results.append(features)

        result_df = pd.DataFrame(results)
        result_df = result_df[['AccessionName'] + [col for col in result_df.columns if col != 'AccessionName']]
        return result_df

    def get_linker_features(self, records, require_complete=True):
        frame = self.process_single_sequence_file(self.apod_file)
        return finalize_features(frame, records, require_complete=require_complete)


if __name__ == "__main__":
    config = dataset_config(args.dataset)
    records = read_optional_label_fasta(config["fasta"])
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    output = output_name(config, "Linker")
    extractor = LinkerFeatureExtractor(config["linker"])
    linker_features_df = extractor.get_linker_features(records, not args.allow_missing)
    linker_features_df.to_csv(output, index=False, encoding="utf-8-sig")
#    print(f"Wrote {len(linker_features_df)} rows to {output}")
