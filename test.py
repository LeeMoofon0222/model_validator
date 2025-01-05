import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import shap
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

def first_page():
    # Create window
    root = tk.Tk()
    root.title("Model Evaluation Tool")

    # Set window size
    window_width = 1200
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_offset = int((screen_width - window_width) / 2)
    y_offset = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")

    # Keep the window always on top
    root.attributes('-topmost', True)

    # Initialize global variables after creating the root window
    selected_model = tk.StringVar()
    selected_target = tk.StringVar()
    selected_value = tk.StringVar()
    selected_metrics = {}
    model_types = ["rf", "xgboost", "lgbm"]
    target_types = ["classification", "numeric"]
    target_values = [0, 1]

    # Functions
    def upload_model():
        filepath = filedialog.askopenfilename(title="Choose model", filetypes=[("Model Files", "*.h5 *.pt *.pkl *.joblib")])
        if filepath:
            global model_path
            model_path = filepath
            #selected_model.set(model_path)
            messagebox.showinfo("Upload model successful", f"Chosen model:\n{filepath}")
            model_type_dropdown.config(state="normal")  # Enable dropdown after uploading model

    def upload_train_data():
        filepath = filedialog.askopenfilename(title="Upload data", filetypes=[("Data Files", "*.csv *.json *.xlsx")])
        if filepath:
            global train_data
            train_data = pd.read_csv(filepath)
            if 'Diabetes_binary' in train_data.columns:
                train_data = train_data.drop('Diabetes_binary', axis=1)
            messagebox.showinfo("Upload data successful", f"Chosen data:\n{filepath}")

            # Update feature list after loading training data
            update_feature_checkbuttons()

    def upload_test_data():
        filepath = filedialog.askopenfilename(title="Choose test data", filetypes=[("Data Files", "*.csv *.json *.xlsx")])
        if filepath:
            global test_data
            test_data = pd.read_csv(filepath)

    def update_feature_checkbuttons():
        if train_data is not None:
            feature_names = list(train_data.columns)
            for widget in right_frame.winfo_children():
                widget.destroy()  # Clear previous widgets

            tk.Label(right_frame, text="Features", font=("Arial", 12, "bold"), bg="#ffffff").pack(anchor="center", pady=5)
            for metric in feature_names:
                var = tk.BooleanVar()
                selected_metrics[metric] = var
                tk.Checkbutton(right_frame, text=metric, variable=var, bg="#ffffff").pack(anchor="w", pady=5)

    def confirm_selection():
        selected = selected_model.get()
        metrics = [metric for metric, value in selected_metrics.items() if value.get()]
        value = selected_value.get() if selected_target.get() == "classification" else "N/A"
        messagebox.showinfo("Selection Confirmation", f"Chosen model: {selected}\nEvaluation metrics: {', '.join(metrics)}\nThreshold: {value}")


    def on_target_change(*args):
        if selected_target.get() == "classification":
            target_value_entry.config(state="normal")
        else:
            target_value_entry.config(state="disabled")
            selected_value.set(target_values[0])  # Reset the threshold selection

    def update_protected_attributes():
        # Clear previous widgets in right_frame2
        for widget in right_frame2.winfo_children():
            widget.destroy()

        if fairness_check.get():  # If Fairness Check is selected
            tk.Label(right_frame2, text="Protected Attributes", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(anchor="center", pady=5)

            if train_data is not None:
                feature_names = list(train_data.columns)
                for feature in feature_names:
                    var = tk.BooleanVar(value=False)  # Default to not selected
                    tk.Checkbutton(right_frame2, text=feature, variable=var, bg="#f0f0f0").pack(anchor="w", padx=10, pady=5)

    # Left and right frames
    left_frame = tk.Frame(root, height=400, width=600, bg="#f0f0f0")
    right_frame = tk.Frame(root, height=800, width=300, bg="#ffffff")
    left_bottom_frame = tk.Frame(root, height=400, width=600, bg="#d0d0d0")  # New frame below left_frame
    right_frame2 = tk.Frame(root, height=800, width=300, bg="#f0f0f0")

    
    right_frame2.pack(side="right", fill="both", padx=10, pady=10)
    right_frame.pack(side="right", fill="both", padx=10, pady=10)
    left_frame.pack(side="top", fill="both", padx=10, pady=10)
    left_bottom_frame.pack(side="bottom", fill="both", padx=10, pady=10)  # Bottom frame packed below left_frame


    
    # Prevent frames from resizing
    left_frame.pack_propagate(False)
    right_frame.pack_propagate(False)
    left_bottom_frame.pack_propagate(False)
    right_frame2.pack_propagate(False)


    # Left frame content
    tk.Label(left_frame, text="Operation Area", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(anchor="center", pady=5)
    btn_upload_model = tk.Button(left_frame, text="Upload model", command=upload_model, width=15)
    btn_upload_model.pack(anchor="center", pady=5)

    btn_upload_train_data = tk.Button(left_frame, text="Upload training data", command=upload_train_data, width=15)
    btn_upload_train_data.pack(anchor="center", pady=5)

    btn_upload_test_data = tk.Button(left_frame, text="Upload testing data", command=upload_test_data, width=15)
    btn_upload_test_data.pack(anchor="center", pady=5)

    # Select model type
    tk.Label(left_frame, text="Choose model types:", bg="#f0f0f0").pack(anchor="center", pady=5)
    model_type_dropdown = tk.OptionMenu(left_frame, selected_model, *model_types)
    model_type_dropdown.config(state="normal")  # Disable dropdown initially
    model_type_dropdown.pack(anchor="center", pady=5)
    selected_model.set(model_types[0])

    #Select  target types
    tk.Label(left_frame, text="Choose target types:", bg="#f0f0f0").pack(anchor="center", pady=5)
    model_dropdown = tk.OptionMenu(left_frame, selected_target, *target_types)
    model_dropdown.pack(anchor="center", pady=5)
    selected_target.set(target_types[0])


    #Enter target value
    tk.Label(left_frame, text="Enter target value:", bg="#f0f0f0").pack(anchor="center", pady=5)
    '''
    target_value_dropdown = tk.OptionMenu(left_frame, selected_value, *target_values)
    target_value_dropdown.config(state="normal")  # Initially disabled
    target_value_dropdown.pack(anchor="center", pady=5)
    selected_value.set(target_values[0])
    '''
    def validate(P):
        #print(P)
        if str.isdigit(P) or P == '':
            return True
        else:
            return False
    vcmd = (root.register(validate), '%P')
    target_value_entry = tk.Entry(left_frame, text = "target value", validate='key', validatecommand=vcmd)
    target_value_entry.pack(anchor="center", pady=5)

    # Bind target change event
    selected_target.trace("w", on_target_change)

    # left bottom frame content
    tk.Label(left_bottom_frame, text="Explain Method", font=("Arial", 12, "bold"), bg="#d0d0d0").pack(anchor="center", pady=5)
    # Initialize variables for checkboxes
    lime = tk.BooleanVar(value=False)
    shap = tk.BooleanVar(value=False)
    quality_check = tk.BooleanVar(value=False)
    fairness_check = tk.BooleanVar(value=False)
    drift_check = tk.BooleanVar(value=False)

    # Create Checkbuttons for each element
    tk.Checkbutton(left_bottom_frame, text="LIME", variable=lime, bg="#d0d0d0").pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(left_bottom_frame, text="SHAP", variable=shap, bg="#d0d0d0").pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(left_bottom_frame, text="Quality Check", variable=quality_check, bg="#d0d0d0").pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(left_bottom_frame, text="Fairness Check", variable=fairness_check, bg="#d0d0d0").pack(anchor="w", padx=10, pady=5)
    tk.Checkbutton(left_bottom_frame, text="Drift Check", variable=drift_check, bg="#d0d0d0").pack(anchor="w", padx=10, pady=5)

    fairness_check.trace("w", lambda *args: update_protected_attributes())

    # Right frame content
    tk.Label(right_frame, text="Features", font=("Arial", 12, "bold"), bg="#ffffff").pack(anchor="center", pady=5)
    # Bottom confirm button
    btn_confirm = tk.Button(left_frame, text="Confirm selection", command=confirm_selection, width=15)
    btn_confirm.pack(side="bottom", anchor="e", pady=15)

    tk.Label(right_frame2, text="Protected attributes", font=("Arial", 12, "bold"), bg="#ffffff").pack(anchor="center", pady=5)

    # Main loop
    root.mainloop()


# Call the function to run the program
first_page()
