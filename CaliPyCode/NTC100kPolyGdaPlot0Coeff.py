def update_model():
    if x is None or y_desired is None:
        messagebox.showwarning("No Data", "Please load data before updating the model.")
        return

    degree = int(degree_slider.get())
    alpha = float(alpha_slider.get())
    
    x_poly = polynomial_features(x, degree)
    X_train, X_test, y_train, y_test = train_test_split_manual(x_poly, y_desired, test_size=0.2, random_state=42)
    beta = ridge_regression(X_train, y_train, alpha)
    y_pred_train = predict(X_train, beta)
    y_pred_test = predict(X_test, beta)

    train_error = percentage_error(y_train, y_pred_train)
    test_error = percentage_error(y_test, y_pred_test)
    
    # Print the coefficients
    print(f"Coefficients (beta) for Polynomial Degree {degree} and Alpha {alpha}:")
    for i, coef in enumerate(beta):
        print(f"  Coefficient for x^{i}: {coef}")
    print("\n")

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_desired, color='blue', label='Actual')
    plt.scatter(x, predict(x_poly, beta), color='red', label='Predicted')
    plt.xlabel('Frequency')
    plt.ylabel('Temperature')
    plt.legend()

    # Update the title to show percentage errors and file name
    file_name = file_path.split('/')[-1] if file_path else 'Unknown File'
    plt.title(f'{file_name} - Train Error: {train_error:.2f}%, Test Error: {test_error:.2f}%')
    
    plt.show()

    save_results(beta, x_poly)
