import customtkinter as ctk
from tkinter import StringVar, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MLModel(ctk.CTk):
    def __init__(self, processed_df=None):
        super().__init__()

        self.title("ML AlgoHub - Dataset Models")
        self.geometry("1200x750+250+50")
        self.processed_df = processed_df
        self.feature_vars = {}
        self.label_encoder = LabelEncoder()

        # Algorithm mapping
        self.algo_map = {
            "Regression": [
                "Linear Regression", "Ridge Regression", "Lasso Regression",
                "SVR", "Decision Tree Regressor", "Random Forest Regressor"
            ],
            "Classification": [
                "Logistic Regression", "Naive Bayes", "Random Forest Classifier",
                "SVC", "KNN", "Decision Tree Classifier"
            ],
            "Clustering": [
                "KMeans", "DBSCAN"
            ]
        }

        self.model_instances = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "SVR": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Logistic Regression": LogisticRegression(max_iter=300),
            "Naive Bayes": GaussianNB(),
            "Random Forest Classifier": RandomForestClassifier(),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }

        # Model parameters
        self.model_params = {
            "KMeans": {"n_clusters": 3},
            "DBSCAN": {"eps": 0.5, "min_samples": 5}
        }

        self.create_gui()

    def create_gui(self):
        # Top Header Frame
        top_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=60)
        top_frame.pack(side="top", fill="x")
        ctk.CTkLabel(top_frame, text="Dataset Models", font=("Arial", 28, "bold"), text_color="#00b7eb").pack(pady=10)

        # Main Body Frame
        body_frame = ctk.CTkFrame(self)
        body_frame.pack(fill="both", expand=True)

        # Sidebar with Vertical Scroll
        sidebar_scroll = ctk.CTkScrollableFrame(body_frame, width=300, fg_color="#1a1a1a")
        sidebar_scroll.pack(side="left", fill="y", padx=10, pady=10)

        # Sidebar Content
        sidebar = ctk.CTkFrame(sidebar_scroll, fg_color="#1a1a1a")
        sidebar.pack(fill="both", expand=True, padx=10, pady=10)

        # Step 1: Model Type Selection
        model_type_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        model_type_frame.pack(fill="x", pady=(10, 5))
        ctk.CTkLabel(model_type_frame, text="Step 1: Select Model Type", font=("Arial", 14, "bold"), text_color="#00b7eb").pack(pady=(5, 10))
        self.model_type_var = StringVar(value="Supervised")
        ctk.CTkRadioButton(
            model_type_frame, text="Supervised", variable=self.model_type_var, value="Supervised",
            font=("Arial", 12), text_color="white", fg_color="#00b7eb", command=self.update_tasks
        ).pack(pady=5)
        ctk.CTkRadioButton(
            model_type_frame, text="Unsupervised", variable=self.model_type_var, value="Unsupervised",
            font=("Arial", 12), text_color="white", fg_color="#00b7eb", command=self.update_tasks
        ).pack(pady=5)

        # Step 2: Task Selection
        task_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        task_frame.pack(fill="x", pady=(10, 5))
        ctk.CTkLabel(task_frame, text="Step 2: Select Task", font=("Arial", 14, "bold"), text_color="#00b7eb").pack(pady=(5, 10))
        self.task_var = StringVar(value="Classification")
        self.task_dropdown = ctk.CTkComboBox(
            task_frame, variable=self.task_var, values=["Classification", "Regression"],
            command=self.update_algorithms, state="readonly", width=200
        )
        self.task_dropdown.pack(pady=5)

        # Step 3: Data Configuration
        data_config_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        data_config_frame.pack(fill="x", pady=(10, 5))
        ctk.CTkLabel(data_config_frame, text="Step 3: Data Configuration", font=("Arial", 14, "bold"), text_color="#00b7eb").pack(pady=(5, 10))

        # Target Column Selection
        self.target_label = ctk.CTkLabel(data_config_frame, text="Select Target Column:", font=("Arial", 12, "bold"))
        self.target_label.pack(pady=(5, 5), anchor="w")
        self.target_var = StringVar()
        columns = list(self.processed_df.columns) if self.processed_df is not None else []
        self.target_dropdown = ctk.CTkComboBox(
            data_config_frame, variable=self.target_var, values=columns, state="readonly", width=200
        )
        self.target_dropdown.pack(pady=5)
        if columns:
            self.target_var.set(columns[0])
            self.target_dropdown.configure(state="readonly")
        else:
            self.target_dropdown.configure(state="disabled")

        # Feature Selection
        ctk.CTkLabel(data_config_frame, text="Select Features:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        feature_frame = ctk.CTkScrollableFrame(data_config_frame, height=150, fg_color="#2b2b2b")
        feature_frame.pack(pady=5, fill="x")
        self.update_features(feature_frame)

        # Feature Selection Buttons
        ctk.CTkButton(data_config_frame, text="Select All Features", command=self.select_all_features, fg_color="#1E3A46").pack(pady=5)
        ctk.CTkButton(data_config_frame, text="Deselect All Features", command=self.deselect_all_features, fg_color="#1E3A46").pack(pady=5)

        # Test Size
        ctk.CTkLabel(data_config_frame, text="Test Size (Default 20%):", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.test_size_entry = ctk.CTkEntry(data_config_frame, width=100, placeholder_text="0.2")
        self.test_size_entry.pack(pady=5)

        # Step 4: Model Selection
        model_select_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        model_select_frame.pack(fill="x", pady=(10, 5))
        ctk.CTkLabel(model_select_frame, text="Step 4: Select Model", font=("Arial", 14, "bold"), text_color="#00b7eb").pack(pady=(5, 10))
        self.model_var = StringVar()
        self.model_dropdown = ctk.CTkOptionMenu(model_select_frame, variable=self.model_var, values=self.algo_map["Classification"])
        self.model_dropdown.pack(pady=5)

        # Model Parameters (Dynamic)
        self.params_frame = ctk.CTkFrame(model_select_frame, fg_color="#2b2b2b")
        self.params_frame.pack(fill="x", pady=5)
        self.update_model_params()

        # Step 5: Run Models
        run_frame = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        run_frame.pack(fill="x", pady=(10, 5))
        ctk.CTkLabel(run_frame, text="Step 5: Run Models", font=("Arial", 14, "bold"), text_color="#00b7eb").pack(pady=(5, 10))
        self.run_all_button = ctk.CTkButton(
            run_frame, text="Apply All Models", fg_color="green", height=50, command=self.run_all_models
        )
        self.run_all_button.pack(pady=10)
        self.submit_button = ctk.CTkButton(run_frame, text="Run Selected Model", fg_color="#00b7eb", command=self.run_selected_model)
        self.submit_button.pack(pady=5)

        # Model Output Area (Dynamic with Scroll)
        self.model_area = ctk.CTkScrollableFrame(body_frame, fg_color="#2b2b2b")
        self.model_area.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.model_frames = {}
        self.result_labels = {}
        self.update_model_output()

        # Initialize
        self.update_tasks()
        self.update_algorithms(None)
        self.validate_inputs()

    def update_tasks(self):
        model_type = self.model_type_var.get()
        tasks = []
        if model_type == "Supervised":
            tasks = ["Classification", "Regression"]
        else:  # Unsupervised
            tasks = ["Clustering"]
        self.task_dropdown.configure(values=tasks)
        self.task_var.set(tasks[0])
        self.update_algorithms(None)

        # Show/Hide Target Selection based on Task
        if model_type == "Unsupervised":
            self.target_label.pack_forget()
            self.target_dropdown.pack_forget()
        else:
            self.target_label.pack(pady=(5, 5), anchor="w")
            self.target_dropdown.pack(pady=5)

    def update_features(self, feature_frame):
        self.feature_vars.clear()
        if self.processed_df is None:
            return
        for col in self.processed_df.columns:
            var = ctk.BooleanVar(value=True)
            self.feature_vars[col] = var
            ctk.CTkCheckBox(feature_frame, text=col, variable=var, command=self.validate_inputs).pack(anchor="w", padx=5, pady=2)

    def select_all_features(self):
        for var in self.feature_vars.values():
            var.set(True)
        self.validate_inputs()

    def deselect_all_features(self):
        for var in self.feature_vars.values():
            var.set(False)
        self.validate_inputs()

    def update_algorithms(self, _):
        task = self.task_var.get()
        algorithms = self.algo_map.get(task, [])
        self.model_var.set(algorithms[0] if algorithms else "")
        self.model_dropdown.configure(values=algorithms)
        self.update_model_params()
        self.validate_inputs()

    def update_model_params(self):
        # Clear previous parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        selected_model = self.model_var.get()
        if selected_model in self.model_params:
            params = self.model_params[selected_model]
            ctk.CTkLabel(self.params_frame, text="Model Parameters:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(5, 2))
            if selected_model == "KMeans":
                ctk.CTkLabel(self.params_frame, text="Number of Clusters:", font=("Arial", 11)).pack(anchor="w", padx=5)
                n_clusters_entry = ctk.CTkEntry(self.params_frame, width=100, placeholder_text="3")
                n_clusters_entry.insert(0, str(params["n_clusters"]))
                n_clusters_entry.pack(anchor="w", padx=5, pady=2)
                n_clusters_entry.bind("<FocusOut>", lambda e: self.update_param(selected_model, "n_clusters", n_clusters_entry.get()))
            elif selected_model == "DBSCAN":
                ctk.CTkLabel(self.params_frame, text="Epsilon (eps):", font=("Arial", 11)).pack(anchor="w", padx=5)
                eps_entry = ctk.CTkEntry(self.params_frame, width=100, placeholder_text="0.5")
                eps_entry.insert(0, str(params["eps"]))
                eps_entry.pack(anchor="w", padx=5, pady=2)
                eps_entry.bind("<FocusOut>", lambda e: self.update_param(selected_model, "eps", eps_entry.get()))

                ctk.CTkLabel(self.params_frame, text="Min Samples:", font=("Arial", 11)).pack(anchor="w", padx=5)
                min_samples_entry = ctk.CTkEntry(self.params_frame, width=100, placeholder_text="5")
                min_samples_entry.insert(0, str(params["min_samples"]))
                min_samples_entry.pack(anchor="w", padx=5, pady=2)
                min_samples_entry.bind("<FocusOut>", lambda e: self.update_param(selected_model, "min_samples", min_samples_entry.get()))

    def update_param(self, model_name, param_name, value):
        try:
            value = float(value) if param_name in ["eps"] else int(value)
            self.model_params[model_name][param_name] = value
            if model_name == "KMeans":
                self.model_instances[model_name] = KMeans(n_clusters=self.model_params[model_name]["n_clusters"], random_state=42)
            elif model_name == "DBSCAN":
                self.model_instances[model_name] = DBSCAN(
                    eps=self.model_params[model_name]["eps"],
                    min_samples=self.model_params[model_name]["min_samples"]
                )
        except ValueError:
            messagebox.showerror("Error", f"Invalid value for {param_name}. Please enter a valid number.")

    def update_model_output(self):
        # Clear previous output
        for widget in self.model_area.winfo_children():
            widget.destroy()
        self.model_frames = {}
        self.result_labels = {}

    def validate_inputs(self):
        valid = True
        error_message = ""
        if self.processed_df is None or self.processed_df.empty:
            valid = False
            error_message = "No dataset loaded."
        else:
            task = self.task_var.get()
            selected_features = [col for col, var in self.feature_vars.items() if var.get()]
            if not selected_features:
                valid = False
                error_message = "No features selected."
            else:
                if task in ["Classification", "Regression"]:
                    target = self.target_var.get()
                    if not target:
                        valid = False
                        error_message = "No target column selected."
                    else:
                        selected_features = [col for col in selected_features if col != target]
                        if task == "Regression":
                            if not pd.api.types.is_numeric_dtype(self.processed_df[target]):
                                valid = False
                                error_message = "Regression requires a numeric target."
                        elif task == "Classification":
                            if pd.api.types.is_numeric_dtype(self.processed_df[target]):
                                unique_values = self.processed_df[target].nunique()
                                if unique_values > 20:
                                    valid = False
                                    error_message = f"Classification target has too many unique values ({unique_values})."
                elif task == "Clustering":
                    if not all(pd.api.types.is_numeric_dtype(self.processed_df[feat]) for feat in selected_features):
                        valid = False
                        error_message = "Clustering requires numeric features."

        self.run_all_button.configure(state="normal" if valid else "disabled")
        self.submit_button.configure(state="normal" if valid else "disabled")
        if not valid and error_message:
            print(f"Validation failed: {error_message}")

    def get_data(self):
        task = self.task_var.get()
        selected_features = [col for col, var in self.feature_vars.items() if var.get()]
        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature.")
            return None, None, None, None

        try:
            test_size = float(self.test_size_entry.get())
        except ValueError:
            test_size = 0.2

        X = self.processed_df[selected_features]
        y = None

        if task in ["Classification", "Regression"]:
            target = self.target_var.get()
            if not target:
                messagebox.showerror("Error", "Please select a target column.")
                return None, None, None, None
            y = self.processed_df[target]
            selected_features = [col for col in selected_features if col != target]

            if task == "Classification" and not pd.api.types.is_numeric_dtype(y):
                try:
                    y = self.label_encoder.fit_transform(y)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to encode target: {str(e)}")
                    return None, None, None, None

        try:
            X = X.select_dtypes(include=[np.number])
            if X.empty:
                messagebox.showerror("Error", "No numeric features selected. Please preprocess features.")
                return None, None, None, None
        except Exception as e:
            messagebox.showerror("Error", f"Feature processing failed: {str(e)}")
            return None, None, None, None

        if task in ["Classification", "Regression"]:
            print(f"Target type: {y.dtype}, Unique values: {np.unique(y)}")
            return train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            return X, None, None, None

    def run_model(self, model_name, model_instance):
        X, _, _, _ = self.get_data()
        if X is None:
            return

        task = self.task_var.get()
        try:
            print(f"Running {model_name} for task: {task}")
            if task in ["Classification", "Regression"]:
                X_train, X_test, y_train, y_test = self.get_data()
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_test)

                if task == "Regression":
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    report = f"R² Score: {r2:.4f}\nMean Squared Error: {mse:.4f}"
                else:
                    acc = accuracy_score(y_test, y_pred)
                    report = f"Accuracy: {acc*100:.2f}%\n{classification_report(y_test, y_pred, digits=2)}"
            else:
                labels = model_instance.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                report = f"Number of clusters: {n_clusters}\nLabels: {np.unique(labels)}"

            # Add result to output area
            frame = ctk.CTkFrame(self.model_area, fg_color="#1e1e1e")
            frame.pack(fill="x", padx=5, pady=5)
            ctk.CTkLabel(frame, text=f"{model_name}", font=("Courier", 14)).pack(padx=10, pady=(10, 2))
            result_label = ctk.CTkLabel(frame, text=report, font=("Courier", 11), justify="left")
            result_label.pack(padx=5, pady=5)
            self.model_frames[model_name] = frame
            self.result_labels[model_name] = result_label
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {model_name}: {str(e)}")
            print(f"Error in {model_name}: {str(e)}")

    def run_selected_model(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        selected_model = self.model_var.get()
        self.update_model_output()  # Clear previous results
        self.run_model(selected_model, self.model_instances[selected_model])
        self.validate_inputs()

    def run_all_models(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        task = self.task_var.get()
        algorithms = self.algo_map[task]
        self.update_model_output()  # Clear previous results
        for model_name in algorithms:
            self.run_model(model_name, self.model_instances[model_name])
        self.validate_inputs()

# Example usage:
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    app = MLModel(processed_df=df)
    app.mainloop()