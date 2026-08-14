import os
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import gnss_forecast as gf

# Disable TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def objective(trial):
    # Suggest hyperparameters
    bilstm_units = trial.suggest_categorical("bilstm_units", [32, 64, 128])
    gru_units = trial.suggest_categorical("gru_units", [16, 32, 64])
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.4)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.4)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    N_FEATURES = len(gf.TARGET_COLS)
    inp = Input(shape=(gf.SEQ_LEN, N_FEATURES), name="input")

    # Encoder
    x = Bidirectional(LSTM(bilstm_units, return_sequences=True), name="bilstm")(inp)
    x = Dropout(dropout_1, name="drop1")(x)
    x = GRU(gru_units, return_sequences=False, name="gru")(x)
    x = Dropout(dropout_2, name="drop2")(x)
    x = LayerNormalization(name="layernorm")(x)

    # Projection
    x = Dense(64, activation="relu", name="dense1")(x)
    out = Dense(gf.FORECAST_HORIZON * N_FEATURES, name="dense_out")(x)
    out = tf.keras.layers.Reshape((gf.FORECAST_HORIZON, N_FEATURES), name="output")(out)

    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=lr), loss="huber", metrics=["mae"])

    global X_tr, y_tr, X_val, y_val

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=15, # 15 epochs max for tuning
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    val_mae = min(history.history["val_mae"])
    return val_mae

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="FINAL_Data.csv")
    args = parser.parse_args()

    print("Loading data for tuning...")
    train_df, test_df, complete_sats = gf.load_and_clean(args.data)
    train_df, test_df, scaler = gf.scale_data(train_df, test_df)
    
    global X_tr, y_tr, X_val, y_val
    X_tr, y_tr, X_val, y_val = gf.build_all_sequences(train_df, complete_sats)

    print(f"Training sequences: {X_tr.shape}, Validation sequences: {X_val.shape}")
    print("Starting Optuna hyperparameter search...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8) # 8 trials to keep execution fast

    print("\n" + "="*50)
    print("OPTUNA TUNING COMPLETE")
    print("="*50)
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Best Val MAE: {trial.value:.5f}")
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
