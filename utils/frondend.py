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
    selected_threshold = tk.StringVar()  # For classification threshold
    selected_metrics = {}
    model_types = ["rf", "xgboost", "lgbm"]
    target_types = ["classification", "numeric"]
    thresholds = ["over 0", "over 1"]

    # Functions
    def upload_model():
        filepath = filedialog.askopenfilename(title="Choose model", filetypes=[("Model Files", "*.h5 *.pt *.pkl *.joblib")])
        if filepath:
            global model_path
            model_path = filepath
            messagebox.showinfo("Upload model successful", f"Chosen model:\n{filepath}")
            model_dropdown.config(state="normal")  # Enable dropdown after uploading model

    def upload_train_data():
        filepath = filedialog.askopenfilename(title="Upload data", filetypes=[("Data Files", "*.csv *.json *.xlsx")])
        if filepath:
            global train_data
            train_data = pd.read_csv(filepath)
            if 'Diabetes_binary' in train_data.columns:
                train_data = train_data.drop('Diabetes_binary', axis=1)
            messagebox.showinfo("Upload data successful", f"Chosen data:\n{filepath}")
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
        threshold = selected_threshold.get() if selected_target.get() == "classification" else "N/A"
        messagebox.showinfo("Selection Confirmation", f"Chosen model: {selected}\nEvaluation metrics: {', '.join(metrics)}\nThreshold: {threshold}")

    def on_target_change(*args):
        if selected_target.get() == "classification":
            threshold_dropdown.config(state="normal")
        else:
            threshold_dropdown.config(state="disabled")
            selected_threshold.set("")  # Reset the threshold selection

    # Bind target change event
    selected_target.trace("w", on_target_change)

    # Left and right frames
    left_frame = tk.Frame(root, width=600, bg="#f0f0f0")
    right_frame = tk.Frame(root, width=600, bg="#ffffff")

    left_frame.pack(side="left", fill="y", padx=10, pady=10)
    right_frame.pack(side="right", fill="y", padx=10, pady=10)

    # Prevent frames from resizing
    left_frame.pack_propagate(False)
    right_frame.pack_propagate(False)

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
    model_dropdown = tk.OptionMenu(left_frame, selected_model, *model_types)
    model_dropdown.config(state="disabled")  # Disable dropdown initially
    model_dropdown.pack(anchor="center", pady=5)
    selected_model.set(model_types[0])

    tk.Label(left_frame, text="Choose target types:", bg="#f0f0f0").pack(anchor="center", pady=5)
    target_dropdown = tk.OptionMenu(left_frame, selected_target, *target_types)
    target_dropdown.pack(anchor="center", pady=5)
    selected_target.set(target_types[0])

    tk.Label(left_frame, text="Choose classification threshold:", bg="#f0f0f0").pack(anchor="center", pady=5)
    threshold_dropdown = tk.OptionMenu(left_frame, selected_threshold, *thresholds)
    threshold_dropdown.config(state="disabled")  # Initially disabled
    threshold_dropdown.pack(anchor="center", pady=5)
    selected_threshold.set(thresholds[0])

    # Right frame content
    tk.Label(right_frame, text="Features", font=("Arial", 12, "bold"), bg="#ffffff").pack(anchor="center", pady=5)

    # Bottom confirm button
    btn_confirm = tk.Button(left_frame, text="Confirm selection", command=confirm_selection, width=15)
    btn_confirm.pack(side="bottom", anchor="e", pady=15)

    # Main loop
    root.mainloop()

# Call the function to run the program
first_page()
