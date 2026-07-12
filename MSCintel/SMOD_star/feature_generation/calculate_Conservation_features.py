import numpy as np
import pandas as pd
import math
import os
from Parameter import args, dataset_config, PROJECT_ROOT
from workflow_utils import read_optional_label_fasta, finalize_features, resolve_project_path
from sequence_feature_common import output_name


#############计算氨基酸的保守性
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
        # 计算 pi*log2(pi/pib)
        return sij * math.log2(sij / pib)
    else:
        return 0


def calculate_relative_entropy_from_hhm(file_path, amino_acid_background):
    # print(f"Processing file: {file_path}")  # 输出当前处理的文件路径
    lines = extract_section(file_path)
    selected_lines = lines[4:][::3]  # 提取第五行及以后，每隔三行取一行数据

    # 解析数据并创建 DataFrame
    df = pd.DataFrame([line.split() for line in selected_lines])

    # 交换第一列和第二列
    df.iloc[:, [0, 1]] = df.iloc[:, [1, 0]].to_numpy()

    # 删除最后一列
    df = df.iloc[:, :-1]

    # 设置列名
    df.columns = ['index', 'AA'] + list(amino_acid_background.keys())

    # 计算每个氨基酸的相对熵值
    for col_index in range(2, df.shape[1]):
        amino_acid = df.columns[col_index]
        pib = amino_acid_background[amino_acid]
        # df.iloc[:, col_index] = df.iloc[:, col_index].apply(lambda sij: calculate_value(sij, pib))
        df[df.columns[col_index]] = df[df.columns[col_index]].apply(lambda sij: calculate_value(sij, pib))

    # 计算相对熵
    df['relative_entropy'] = df.iloc[:, 2:].sum(axis=1)

    return df[['index', 'relative_entropy']]  # 仅保留索引和相对熵



def merge_relative_entropy_data(input_folder):
    all_column_data = {}
    max_length = 0

    # 定义 20 种氨基酸的背景频率
    amino_acid_background = {
        "A": 0.0834, "C": 0.0157, "D": 0.0541, "E": 0.061, "F": 0.0384,
        "G": 0.0668, "H": 0.0228, "I": 0.0533, "K": 0.052, "L": 0.0925,
        "M": 0.0218, "N": 0.0428, "P": 0.0523, "Q": 0.0401, "R": 0.0598,
        "S": 0.079, "T": 0.0583, "V": 0.0631, "W": 0.0127, "Y": 0.0287
    }

    # 遍历所有 .hhm 文件并处理
    for file_name in filter(lambda f: f.endswith('.hhm'), os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file_name)
        df = calculate_relative_entropy_from_hhm(file_path, amino_acid_background)

        column_name = file_name.split('.')[0]  # 提取文件名前缀作为列名
        all_column_data[column_name] = df['relative_entropy'].tolist()

        max_length = max(max_length, len(all_column_data[column_name]))  # 找到最长的列

    # 统一填充较短的列
    for column_name in all_column_data:
        current_length = len(all_column_data[column_name])
        if current_length < max_length:
            # all_column_data[column_name] += [0] * (max_length - current_length)
            all_column_data[column_name] += [np.nan] * (max_length - current_length)

    # 创建合并的 DataFrame
    result_df = pd.DataFrame(all_column_data)

    return result_df

############计算Linker片段保守性特征
# 定义一个函数来处理.apod文件并提取所需信息
def extract_info_from_apod(apod_file_path):
    with open(apod_file_path, 'r') as file:
        lines = [line.rstrip('\n') for line in file]  # 保留原始换行符外的内容

    extracted_info = []
    current_accession = None
    current_data_lines = []  # 存储当前组的数据行

    for line in lines:
        if line.startswith('>'):
            # 遇到新组时，先处理前一个组的数据
            if current_accession is not None:
                # 确保有足够的数据行（至少2行）
                if len(current_data_lines) >= 2:
                    binary_value = current_data_lines[1]
                    extracted_info.append((current_accession, binary_value))

            # 初始化新组
            current_accession = line[1:].strip()  # 去掉">"符号并清理空白
            current_data_lines = []
        else:
            # 非">"开头的行作为数据行，保留原始内容（包括空行）
            current_data_lines.append(line.strip())

    # 处理最后一组数据
    if current_accession is not None and len(current_data_lines) >= 2:
        binary_value = current_data_lines[1]
        extracted_info.append((current_accession, binary_value))

    return extracted_info


# 把1片段长度小于4的片段都转化为0
def replace_short_segments(sequence):
    new_sequence = ''
    current_length = 0

    for char in sequence:
        if char == '1':
            current_length += 1
        else:
            if current_length > 0:
                # 当遇到非'1'字符时，检查当前片段长度是否小于4
                if current_length < 4:
                    new_sequence += '0' * current_length  # 替换短片段中的'1'为'0'
                else:
                    new_sequence += '1' * current_length  # 保留长片段
                current_length = 0
            new_sequence += char  # 添加非'1'字符

    # 处理最后一个片段
    if current_length > 0:
        if current_length < 4:
            new_sequence += '0' * current_length  # 替换短片段中的'1'为'0'
        else:
            new_sequence += '1' * current_length  # 保留长片段

    return new_sequence


# 处理单个.apod文件并返回替换后的数据
def process_single_apod_file(apod_file_path):
    extracted_info = extract_info_from_apod(apod_file_path)  # 提取信息

    # 对每个提取的二进制序列进行替换
    processed_data = []
    for accession_name, binary_value in extracted_info:
        new_binary_value = replace_short_segments(binary_value)  # 处理二进制序列
        processed_data.append((accession_name, new_binary_value))  # 保存处理后的数据

    return processed_data


def processed_data_to_dataframe(processed_data):
    # 将 processed_data 转换为字典，列名为 accession_name，列的数据为二进制值
    data_dict = {}
    max_length = 0

    # 处理每个访问名及其对应的二进制值
    for accession_name, binary_value in processed_data:
        data_dict[accession_name] = list(binary_value)  # 将二进制值转换为列表形式
        max_length = max(max_length, len(binary_value))  # 找到最长的二进制序列

    # 填充较短的二进制序列到相同的长度
    for accession_name in data_dict:
        current_length = len(data_dict[accession_name])
        if current_length < max_length:
            data_dict[accession_name] += [np.nan] * (max_length - current_length)  # 用0填充

    # 创建 DataFrame
    df = pd.DataFrame(data_dict)
    df = df[sorted(df.columns, reverse=True)]

    return df


######################################################处理disorder数据
def classify_amino_acids(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行

    # 去除注释和标题行
    lines = [line.strip() for line in lines if not line.startswith('#') and not line.startswith('Num')]

    sequence = ""
    classifications = []

    # 读取数据并进行初步分类
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        score = float(parts[2])
        classification = 1 if score > 0.576 else 0  # 0/1 分类
        classifications.append(classification)
        sequence += str(classification)

    # 处理连续 1 片段长度小于 4 的情况
    processed_sequence = list(sequence)
    i = 0
    while i < len(processed_sequence):
        if processed_sequence[i] == '1':
            start = i
            while i < len(processed_sequence) and processed_sequence[i] == '1':
                i += 1
            end = i
            if end - start < 4:  # 小于 4 的片段改为 0
                for j in range(start, end):
                    processed_sequence[j] = '0'
        else:
            i += 1

    # 将最终的分类结果转换为整数列表
    final_classifications = [int(c) for c in processed_sequence]

    return final_classifications


def process_folder(folder_path):
    all_data = {}  # 存储所有文件的分类数据
    max_length = 0  # 记录最长的分类序列长度

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            classifications = classify_amino_acids(file_path)

            column_name = os.path.splitext(filename)[0]  # 以文件名作为列名
            all_data[column_name] = classifications

            max_length = max(max_length, len(classifications))  # 记录最长的列长度

    # 统一填充较短的列，使所有列长度相同
    for column_name in all_data:
        current_length = len(all_data[column_name])
        if current_length < max_length:
            all_data[column_name] += [np.nan] * (max_length - current_length)  # 用 NaN 填充短列

    # 创建 DataFrame
    result_df = pd.DataFrame(all_data)

    return result_df



#######################################################################处理SS数据
def extract_second_column_to_dataframe(root_folder):
    """
    遍历 root_folder 下的每个子文件夹，提取 seq.phipsi 文件的第二列SS列，
    以文件夹名称作为列名，合并为一个 DataFrame，并自动填充缺失值
    """
    data_frames = []  # 存储所有 DataFrame
    column_names = []  # 存储列名（文件夹名）

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file == "seq.phipsi":
                file_path = os.path.join(subdir, file)
                folder_name = os.path.basename(subdir)  # 获取文件夹名称

                try:
                    # 读取 seq.phipsi 文件
                    df = pd.read_csv(file_path, delim_whitespace=True, header=None, on_bad_lines="skip")

                    # 确保文件至少有两列
                    if df.shape[1] > 1:
                        second_column = df.iloc[:, 2]  # 提取第二列
                        data_frames.append(pd.DataFrame(second_column))  # 转为 DataFrame
                        column_names.append(folder_name)  # 记录列名

                    else:
                        print(f"文件 {file_path} 列数不足，跳过！")

                except Exception as e:
                    print(f"处理 {file_path} 出错: {e}")

    # 合并 DataFrame，按行数最多的对齐，自动填充 NaN
    result_df = pd.concat(data_frames, axis=1, ignore_index=True)

    # 设置列名（文件夹名称）
    result_df.columns = column_names

    return result_df


def process_ss_column(ss_list):
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

def align_dataframes(df1, df2):
    common_columns = list(set(df1.columns) & set(df2.columns))  # 找出共同的列名
    common_columns.sort()  # 对共同列名进行排序，可根据需求修改排序规则

    # 使用字典来保存新的列
    new_columns_dict = {}

    for column in common_columns:
        new_columns_dict[column + '_1'] = df1[column]  # 将 df1 的列添加到字典中
        new_columns_dict[column + '_2'] = df2[column]  # 将 df2 的列添加到字典中

    # 将字典中的列一次性拼接到结果 DataFrame 中
    result_df = pd.DataFrame(new_columns_dict)


    return result_df

def calculate_means(grouped_df, mode="SS"):
    """
    计算二级结构 (SS) 或 0/1 变量为 1 时的均值。

    参数:
    grouped_df: DataFrame, 包含要计算的列
    mode: str, "SS" 表示计算 H、E、C 的均值，
               "Linker" 或 "Disorder" 表示计算 0/1 变量为 1 时的均值。

    返回:
    DataFrame，包含对应模式的计算结果。
    """
    if mode == "SS":
        categories = ['H', 'E', 'C']
        result = {f"{cat}_Conservation": [] for cat in categories}

        for i in range(0, len(grouped_df.columns), 2):
            col1 = grouped_df.iloc[:, i].astype(str)  # SS列
            col2 = grouped_df.iloc[:, i + 1]  # 数值列

            for cat in categories:
                result[f"{cat}_Conservation"].append(col2[col1 == cat].mean())

    elif mode in ["Linker", "Disorder"]:
        result = {f"{mode}_Conservation": []}

        for i in range(0, len(grouped_df.columns), 2):
            col1 = grouped_df.iloc[:, i].fillna(0).astype(int)  # 0/1 变量
            col2 = grouped_df.iloc[:, i + 1]  # 数值列
            result[f"{mode}_Conservation"].append(col2[col1 == 1].mean())

    else:
        raise ValueError("mode 参数必须是 'SS'、'Linker' 或 'Disorder'")

    result = pd.DataFrame(result)
    result["AccessionName"] = [grouped_df.columns[i][:-2] for i in range(0, len(grouped_df.columns), 2)]
    result = result[["AccessionName"] + [col for col in result.columns if col != "AccessionName"]]
    result.reset_index(drop=True, inplace=True)

    return result

def Get_Conservation_features(hhm_file_path, linker_file_path, disorder_folder_path, SS_folder_path):
    # 计算氨基酸的保守性,dataframe
    relative_entropy_df = merge_relative_entropy_data(hhm_file_path)

    #计算序列的保守性
    # relative_entropy_df_mean = relative_entropy_df.mean(axis=0)
    relative_entropy_df_mean = relative_entropy_df.mean(axis=0, skipna=True)
    relative_entropy_df_mean = relative_entropy_df_mean.reset_index()
    relative_entropy_df_mean.columns = ["AccessionName", "sequence_Conservation"]

    #计算linker的保守性
    linker_processed_data = process_single_apod_file(linker_file_path)
    linker_processed_df = processed_data_to_dataframe(linker_processed_data)
    Linker_aligned_df = align_dataframes(linker_processed_df, relative_entropy_df)
    linker_summary_df = calculate_means(Linker_aligned_df, mode="Linker")

    #计算disorder的保守性
    disorder_processed_df = process_folder(disorder_folder_path)
    Disorder_aligned_df = align_dataframes(disorder_processed_df, relative_entropy_df)
    disorder_summary_df = calculate_means(Disorder_aligned_df, mode="Disorder")

    #计算SS的保守性
    df = extract_second_column_to_dataframe(SS_folder_path)
    df = df.iloc[1:].reset_index(drop=True)
    SS_processed_df = df.apply(lambda col: process_ss_column(col.tolist()), axis=0)
    SS_aligned_df = align_dataframes(SS_processed_df, relative_entropy_df)
    SS_summary_df = calculate_means(SS_aligned_df, mode="SS")

    ##合并特征
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

def Merge_Features(positive_input_folder, positive_apod_file_path, positive_disorder_folder_path, positive_root_folder,
                   negative_input_folder, negative_apod_file_path,negative_disorder_folder_path, negative_root_folder
                   ):
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
    result = get_manifest_conservation(
        config["manifest"], config["linker"], records, not args.allow_missing
    )
    result.to_csv(output, index=False, encoding="utf-8-sig")
#    print(f"Wrote {len(result)} rows to {output}")
