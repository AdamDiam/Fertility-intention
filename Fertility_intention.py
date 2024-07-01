import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.model_selection import GridSearchCV


def preprocess(df):
    # Classify the childwish answers
    mapping = {
        "Probably so": 2,
        "Absolutely so": 2,
        "Probably not": 1,
        "Absolutely not": 1,
        "I don't know": 0
    }

    df['childwish'] = df['childwish'].replace(mapping)

    # Predictor variables
    ## Ego variables
    ego_vars = ["age", "educ_bin", "net_income", "partner_num", "child_num"]

    ## Network composition variables
    tie_vars_imp = [
        "mean_closeness_kin", "mean_closeness_friends",
        "mean_closeness_has_kid", "mean_closeness_want_kid", "mean_closeness_wants_no_kid",
        "mean_closeness_help", "mean_closeness_talk",
        "mean_f2f_kin", "mean_f2f_friends",
        "mean_f2f_has_kid", "mean_f2f_want_kid", "mean_f2f_wants_no_kid",
        "mean_f2f_help", "mean_f2f_talk",
        "mean_nonf2f_kin", "mean_nonf2f_friends",
        "mean_nonf2f_has_kid", "mean_nonf2f_want_kid", "mean_nonf2f_wants_no_kid",
        "mean_nonf2f_help", "mean_nonf2f_talk"]

    tie_vars_imp_cor = [var + "_cor" for var in tie_vars_imp]

    comp_vars_combined = ["no_women", "no_older", "no_high_edu",
                          "no_kin", "no_friends", "no_has_child",
                          "no_child_total", "no_child_u5", "no_child_less_happy",
                          "no_wants_child", "no_wants_no_child", "no_help", "no_talk", "mean_closeness", "mean_f2f",
                          "mean_nonf2f"] + tie_vars_imp_cor

    ## Network structure variables
    dens_vars_imp = ["density_kin", "density_friends", "density_children",
                     "density_wantschildren", "density_childfree", "density_talk",
                     "density_help"]
    dens_vars_imp_cor = [var + "_cor" for var in dens_vars_imp]
    struc_vars_combined = ["comm_1or2", "comm_3orhigher", "modularity", "comp_largest",
                           "diameter", "between_centr", "degree_centr", "avg_betweenness",
                           "avg_closeness", "avg_eigenv", "cliques", "components", "density"] + dens_vars_imp_cor

    # Keep only the attributes that we need
    columns_to_keep = ["childwish"] + ego_vars + comp_vars_combined + struc_vars_combined
    df = df[columns_to_keep]

    # Give the mean value for each NA
    for column in df:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=['educ_bin'])  # 1 for higher education, 0 for lower education
    df['educ_bin_0'] = df['educ_bin_0'].astype(int)
    df['educ_bin_1'] = df['educ_bin_1'].astype(int)

    return df


def evaluate_model(df, feature_names, subgroup_name):
    labels = df['childwish']
    features = df[feature_names]

    # Split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20,
                                                                                random_state=42)

    # Setup the XGBoost and RFECV models
    base_model = XGBClassifier(use_label_encoder=False, random_state=42)
    rfecv = RFECV(estimator=base_model, step=1, cv=5, scoring='f1_macro')

    # Define parameter grid for RFECV and XGBoost
    param_grid_rfecv = {
        'rfecv__estimator__n_estimators': [50, 100],
        'rfecv__estimator__max_depth': [3, 5],
        'rfecv__estimator__learning_rate': [0.01, 0.1],
        'rfecv__min_features_to_select': [5, 10]
    }

    # Define the pipeline including SMOTE and RFECV
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rfecv', rfecv)
    ])

    # Setup GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grid_rfecv, scoring='f1_macro', cv=5)

    try:
        grid_search.fit(train_features, train_labels)
    except Exception as e:
        print(f"An error occurred during GridSearchCV fitting for subgroup {subgroup_name}: {e}")
        return

    best_features = train_features.columns[grid_search.best_estimator_.named_steps['rfecv'].support_]
    train_features = train_features[best_features]
    test_features = test_features[best_features]

    # Setup the EBM
    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(train_features, train_labels)

    # Display the EBM model's global explanation
    ebm_global = ebm.explain_global()
    show(ebm_global)

    # Extract feature importances from EBM
    ebm_importances = ebm_global.data()['scores']
    ebm_feature_names = ebm_global.data()['names']

    return ebm_feature_names, ebm_importances


def get_subgroups(df):
    mean_age = round(df['age'].mean())
    subgroups = {
        'child_num': {
            'child_num >= 1': df[df['child_num'] >= 1],
            'child_num < 1': df[df['child_num'] < 1]
        },
        'age': {
            'Age > 29': df[df['age'] > mean_age],
            'Age <= 29': df[df['age'] <= mean_age]
        },
        'partner_num': {
            'Partner number = 0': df[df['partner_num'] == 0],
            'Partner number = 1': df[df['partner_num'] == 1]
        },
        'net_income': {
            'Net income > 1000': df[df['net_income'] > 1000],
            'Net income <= 1000': df[df['net_income'] <= 1000]
        },
        'educ_bin': {
            'Education Bins = 0': df[df['educ_bin_0'] == 0],
            'Education Bins = 1': df[df['educ_bin_0'] == 1]
        }
    }
    return subgroups


def create_individual_plots(subgroup_results, top_n=10):
    for group_name, subgroup_data in subgroup_results.items():
        combined_importances = {}
        for subgroup_name, (feature_names, importances) in subgroup_data.items():
            for feature, importance in zip(feature_names, importances):
                if feature not in combined_importances:
                    combined_importances[feature] = []
                combined_importances[feature].append((subgroup_name, importance))

        plot_data = []
        for feature, values in combined_importances.items():
            for subgroup_name, importance in values:
                plot_data.append([feature, subgroup_name, importance])

        plot_df = pd.DataFrame(plot_data, columns=['Feature', 'Subgroup', 'Importance'])

        top_features = plot_df.groupby('Feature')['Importance'].mean().nlargest(top_n).index
        plot_df = plot_df[plot_df['Feature'].isin(top_features)]

        plt.figure(figsize=(15, 10))
        sns.set_style("whitegrid")
        palette = sns.color_palette("viridis", n_colors=len(subgroup_data))
        sns.barplot(x='Importance', y='Feature', hue='Subgroup', data=plot_df, palette=palette)
        plt.title(f'EBM Feature Importances for {group_name.capitalize()} Subgroups', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.legend(title='Subgroup', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def create_aggregated_importance_plot(subgroup_results, top_n=10):
    combined_importances = {}
    for group_name, subgroup_data in subgroup_results.items():
        for subgroup_name, (feature_names, importances) in subgroup_data.items():
            for feature, importance in zip(feature_names, importances):
                if feature not in combined_importances:
                    combined_importances[feature] = []
                combined_importances[feature].append(importance)

    # Calculate mean importance for each feature across all subgroups
    aggregated_importances = {feature: np.mean(importances) for feature, importances in combined_importances.items()}

    # Select the top 10 features based on aggregated importance
    top_features = sorted(aggregated_importances, key=aggregated_importances.get, reverse=True)[:top_n]
    top_importances = [aggregated_importances[feature] for feature in top_features]

    # Plot the top 10 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f'Top {top_n} Most Significant Variables Across All Subgroups', fontsize=16)
    plt.xlabel('Aggregated Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.show()

df = pd.read_csv('/Users/adam/Desktop/Thesis/Previous_research/data_subset.csv')
# Exploratory results
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Histogram
bin_edges = range(int(df['age'].min()), int(df['age'].max()) + 2, 1)  # Adjust the step as needed
sns.set_style("whitegrid")
plt.figure(figsize=(14, 7))
sns.histplot(df['age'], bins=bin_edges, kde=True, color='skyblue')
plt.title('Distribution of Age', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(ticks=bin_edges, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.show()

# Boxplot
order = ['Absolutely so', 'Probably so', "I don't know", 'Probably not', 'Absolutely not']
sns.set_style("whitegrid")
plt.figure(figsize=(14, 7))
sns.boxplot(x='childwish', y='age', data=df, palette='pastel', order=order)
plt.title('Fertility Preferences by Age', fontsize=16, fontweight='bold')
plt.xlabel('Fertility Preferences', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.show()
print('The shape of our features is:', df.shape)
df = preprocess(df)
subgroups = get_subgroups(df)

subgroup_results = {}
for group_name, subgroups in subgroups.items():
    subgroup_results[group_name] = {}
    for subgroup_name, subgroup_data in subgroups.items():
        if not subgroup_data.empty:
            feature_names, importances = evaluate_model(subgroup_data, df.columns.difference(['childwish']),
                                                        subgroup_name)
            subgroup_results[group_name][subgroup_name] = (feature_names, importances)

create_individual_plots(subgroup_results, top_n=10)
create_aggregated_importance_plot(subgroup_results, top_n=10)
