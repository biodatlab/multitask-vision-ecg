# Column names should be lowercase.
path_column_name = "path"
filename_column_name = "file_name"
run_number_column_name = "run_num"
train_column_name = "train_80_percent"
dev_column_name = "develop_10_percent"

scar_label_column_name = "scar_cad"
lvef_label_column_name = "lvef"
age_column_name = "age"
year_column_name = "year"
split_column_name = "split"
lvef_40_column_name = "lvef_40"

cut_column_name = "cut"
impute_column_name = "impute"

categorical_feature_column_names = ["female_gender", "dm", "ht", "smoke", "dlp"]
numerical_feature_column_names = ["age"]
imputed_feature_column_names = ["dm", "ht", "smoke", "dlp"]

COLUMN_RENAME_MAP = {
    "age": "age",
    "female": "female_gender",
    "diabetes": "dm",
    "hypertension": "ht",
    "smoke": "smoke",
    "dyslipidemia": "dlp",
}
