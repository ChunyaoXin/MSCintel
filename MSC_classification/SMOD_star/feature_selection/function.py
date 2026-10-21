import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import anderson, pearsonr, ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import clone

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)


def perform_statistical_tests(vectors1, vectors2, columns_of_interest, alpha=0.05):
    significant_keys = []
    p_values_dict = {}

    for key in columns_of_interest:
        data1_values = vectors1[key]
        data2_values = vectors2[key]

        if len(data1_values) < 3 or len(data2_values) < 3:
            print(f"Skip {key}: Insufficient sample size (Group1={len(data1_values)}, Group2={len(data2_values)})")
            continue
        if np.std(data1_values) == 0 or np.std(data2_values) == 0:
            print(f"Skip {key}: Zero variance (Group1_std={np.std(data1_values):.2f}, Group2_std={np.std(data2_values):.2f})")
            continue

        A2_stat1, critical_values1 = anderson(data1_values, dist='norm')[:2]
        A2_stat2, critical_values2 = anderson(data2_values, dist='norm')[:2]

        is_normal1 = A2_stat1 < critical_values1[2]
        is_normal2 = A2_stat2 < critical_values2[2]

        try:
            if is_normal1 and is_normal2:
                t_stat, p_val = ttest_ind(data1_values, data2_values, equal_var=True)
                test_type = "t-test"
            else:
                _, p_val = mannwhitneyu(data1_values, data2_values)
                test_type = "Mann-Whitney U"

            if p_val < alpha:
                significant_keys.append(key)
                p_values_dict[key] = p_val

        except Exception as e:
            print(f"test {key} error: {str(e)}")

    return significant_keys, p_values_dict

def read_all_aaindex_features(file_path):
    record_dict = {} 
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("record_code"):
                record_code = line.split(",")[0].split(": ")[1]

                value_str = line.split("value: ")[1]
                value_list = list(map(float, value_str.split(", ")))

                record_dict[record_code] = value_list

    return record_dict

def filter_and_sort_by_p_value(p_value_dict, file_path):
    filter_dict = read_all_aaindex_features(file_path)
    
    filtered_features = [feature for feature in p_value_dict.keys() if feature in filter_dict.keys()]
    sorted_filtered_features = sorted(filtered_features, key=lambda feature: p_value_dict[feature])
    
    output_dict = {}
    
    for feature in sorted_filtered_features:
        output_dict[feature] = filter_dict[feature]
        
    return output_dict

def select_non_redundant_indices(accession_dict, pcc):
    if not accession_dict: 
        print("No highly repeated metrics generated in this run.")
        return []

    accession_numbers = list(accession_dict.keys())
    merged_vectors = list(accession_dict.values())

    selected_indices = [accession_numbers[0]] 
    selected_vectors = [np.array(merged_vectors[0], dtype=float)]
    merged_vectors = [np.array(vector).astype(float) for vector in merged_vectors]

    for i in range(1, len(accession_numbers)):
        if all(abs(pearsonr(merged_vectors[i], selected_vectors[j])[0]) < pcc for j in range(len(selected_indices))):
            selected_indices.append(accession_numbers[i])
            selected_vectors.append(merged_vectors[i])

    return selected_indices

def perform_iterations_and_analyze(df_positive, df_negative, columns_of_interest, outer_iterations,
                                   inter_iterations, repeat_times, alpha, file_path, pcc):
    positive_total_rows = len(df_positive)
    negative_total_rows = len(df_negative)
    rows_to_select = positive_total_rows // 2
    frequent_keys_list = []
    p_values_collection = {}

    for iteration in tqdm(range(outer_iterations), desc="sampling..."):
        key_count = {key: 0 for key in columns_of_interest}

        for i in range(inter_iterations):
            positive_selected_indices = np.random.choice(positive_total_rows, size=rows_to_select, replace=False)
            negative_selected_indices = np.random.choice(negative_total_rows, size=rows_to_select, replace=False)

            df_positive_selected = df_positive.iloc[positive_selected_indices]
            df_negative_selected = df_negative.iloc[negative_selected_indices]

            vectors1 = {column: df_positive_selected[column].values for column in columns_of_interest}
            vectors2 = {column: df_negative_selected[column].values for column in columns_of_interest}

            significant_results, p_values_dict = perform_statistical_tests(vectors1, vectors2, columns_of_interest, alpha=alpha)

            for key, p_val in p_values_dict.items():
                p_values_collection[key] = p_values_collection.get(key, []) + [p_val]
                if key in significant_results:
                    key_count[key] += 1

        frequently_appearing_keys = {key: count for key, count in key_count.items() if count > repeat_times * inter_iterations}
        frequent_keys_list.append(frequently_appearing_keys)


    all_frequent_keys = set()
    for frequent_keys in frequent_keys_list:
        for key in frequent_keys:
            all_frequent_keys.add(key)

    features_and_average_p_values = {key: np.mean(p_values) for key, p_values in p_values_collection.items() if key in all_frequent_keys}


    if not all_frequent_keys:
        print("No frequent keys found. Please check the threshold and iteration settings.")
        return pd.DataFrame() 

    sorted_features_and_average_p_values = sorted(features_and_average_p_values.items(), key=lambda item: item[1])

    sorted_features_and_average_p_values = dict(sorted_features_and_average_p_values)

    filtered_sorted_features = filter_and_sort_by_p_value(sorted_features_and_average_p_values, file_path)

    if not filtered_sorted_features: 
        print("The filtered dictionary is empty. Outputting significant features directly.")
        return list(sorted_features_and_average_p_values.keys())
    else:  
        print("The filtered dictionary is not empty. Selecting non-redundant AAindex features.")
        non_redundant_indices = select_non_redundant_indices(filtered_sorted_features, pcc)

        return non_redundant_indices

def _subset_rows(data, indices):
    """Subset pandas DataFrame/Series or numpy array by row indices."""
    return data.iloc[indices] if hasattr(data, "iloc") else data[indices]


def calculate_average_auc_aps(model, X, y, cv):
    """
    Re-evaluate the selected feature subset using the same CV splitter.

    Parameters
    ----------
    model : estimator
        Machine-learning model.

    X : pandas.DataFrame
        Feature matrix after feature selection.

    y : pandas.Series or numpy.ndarray
        Binary labels.

    cv : cross-validation splitter
        The same StratifiedKFold object used in SFFS.

    Returns
    -------
    average_auc : float
        Mean ROC-AUC across CV folds.

    average_aps : float
        Mean average precision / AUPRC across CV folds.
    """
    auc_scores = []
    aps_scores = []

    for train_index, test_index in cv.split(X, y):
        fold_model = clone(model)

        X_train = _subset_rows(X, train_index)
        X_test = _subset_rows(X, test_index)
        y_train = _subset_rows(y, train_index)
        y_test = _subset_rows(y, test_index)

        fold_model.fit(X_train, y_train)
        y_pred_proba = fold_model.predict_proba(X_test)[:, 1]

        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        aps_scores.append(average_precision_score(y_test, y_pred_proba))

    average_auc = float(np.mean(auc_scores))
    average_aps = float(np.mean(aps_scores))

    return average_auc, average_aps


def get_wrapper_models():
    """Return models used for wrapper feature selection."""
    return {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

def perform_feature_selection_and_auc_calculation(X, y, significant_features, cv, models=None, progress_callback=None):
    if models is None:
        models = get_wrapper_models()

    selected_features_list = []
    selection_auc_list = []
    refit_auc_list = []
    refit_aps_list = []

    for model_name, model in tqdm(models.items(), desc="model test", file=sys.stdout, leave=False):
        if progress_callback:
            progress_callback()

        sffs = SFS(
            estimator=model,
            k_features=(1, len(significant_features)),
            forward=True,
            floating=True,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        with tqdm(total=4, desc=f"{model_name} features select", leave=False) as sub_pbar:
            sffs.fit(X, y)
            sub_pbar.update(1)

            selected_features = list(sffs.k_feature_names_)
            sub_pbar.update(1)

            # AUC obtained directly from the SFFS selection process.
            selection_cv_auc = float(sffs.k_score_)

            # Re-evaluate the selected feature subset using the same CV splitter.
            refit_cv_auc, refit_cv_aps = calculate_average_auc_aps(model=model, X=X[selected_features], y=y, cv=cv)
            sub_pbar.update(1)

            selected_features_list.append((model_name, selected_features))
            selection_auc_list.append((model_name, selection_cv_auc))
            refit_auc_list.append((model_name, refit_cv_auc))
            refit_aps_list.append((model_name, refit_cv_aps))
            sub_pbar.update(1)

        if progress_callback:
            progress_callback()

    return selected_features_list, selection_auc_list, refit_auc_list, refit_aps_list

def normalize(df):
    """Calculate column-wise mean and standard deviation of the DataFrame, then standardize the DataFrame."""
    df = df.fillna(df.mean())
    mean = df.mean()
    std = df.std()
    scaled_df = (df - mean) / std
    return scaled_df, mean, std
