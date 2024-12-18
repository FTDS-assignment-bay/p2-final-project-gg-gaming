# ================== Model Building ==================

def build_ann(input_dim: int):
    """Build and compile an ANN model with callbacks."""
    # Import necessary TensorFlow components
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.initializers import HeNormal
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Clear session
    tf.keras.backend.clear_session()
    np.random.seed(0)
    tf.random.set_seed(0)
    
    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(365, activation='relu'),
        Dense(127, activation='relu'),
        Dense(30, activation='relu', kernel_initializer=HeNormal(seed=0)),
        Dense(17, activation='relu', kernel_initializer=HeNormal(seed=0)),
        Dense(16, activation='relu', kernel_initializer=HeNormal(seed=0)),
        Dense(8, activation='relu', kernel_initializer=HeNormal(seed=0)),
        Dense(2, activation='relu', kernel_initializer=HeNormal(seed=0)),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0), metrics=['mae'])
    
    # Callbacks inside the function
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    return model, callbacks

# ================== Main Modeling Process ==================

def Modeling():
    """Main function to train and evaluate the model."""
    # Import necessary libraries
    from sklearn.metrics import r2_score, mean_absolute_error
    import pickle

    # Load Data
    with open("/opt/airflow/data/data_after_fe.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    X_train_scaled = loaded_data["X_train_scaled"]
    X_val_scaled = loaded_data["X_val_scaled"]
    X_test_scaled = loaded_data["X_test_scaled"]
    y_train = loaded_data["y_train"]
    y_val = loaded_data["y_val"]
    y_test = loaded_data["y_test"]

    # Build Model
    input_dim = X_train_scaled.shape[1]
    model, callbacks = build_ann(input_dim)

    # Train the Model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100, batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Predict and Evaluate
    y_pred_val = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2 Score): {r2}")

    # Evaluate model
    y_pred_test = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2 Score): {r2:.2f}")

# ================== Entry Point ==================

if __name__ == "__main__":
    Modeling()
