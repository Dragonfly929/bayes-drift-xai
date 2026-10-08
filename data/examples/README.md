# Example CSVs for the app's "Upload your own CSV" option

| file | rows | label column | columns to ignore | origin |
|---|---|---|---|---|
| `breast_cancer.csv` | 569 | `target` (0/1) | — | Wisconsin Diagnostic Breast Cancer, UCI id 17 (CC BY 4.0), via scikit-learn |
| `wine.csv` | 178 | `target` (3 classes) | — | Wine, UCI id 109 (CC BY 4.0), via scikit-learn — very small, results are noisy |
| `messy_clients.csv` | 3 000 | `bad` (yes/no, 30 missing) | `client_id`, `note` (pre-selected by the app) | synthetic, generated with seed 0 |

None of these were used in the paper; they exist to check that the pipeline runs on data it has never seen.
