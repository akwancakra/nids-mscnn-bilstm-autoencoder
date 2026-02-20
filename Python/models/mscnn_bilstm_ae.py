import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np

class AttentionLayer(layers.Layer):
    """
    Bahdanau-style Attention Mechanism for LSTM output.
    Allows the model to focus on important timesteps.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, steps, features)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, steps, features)
        # e = tanh(dot(x, W) + b)
        e = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # a = softmax(e) -> alignment scores
        a = tf.nn.softmax(e, axis=1)
        # output = x * a -> weighted sum context
        output = x * a
        # Sum over timesteps if we want a single vector, but for AE we often keep sequence or sum.
        # Here we return the weighted sequence for further processing or the context vector.
        # For this specific architecture, we might want to keep the sequence form if we feed into another LSTM,
        # or reduce if we go to latent. Let's return the context vector (sum).
        context_vector = tf.reduce_sum(output, axis=1)
        return context_vector

def build_multiscale_cnn_bilstm_ae(input_shape, encoding_dim=32):
    """
    Constructs a Multi-Scale CNN-BiLSTM Autoencoder.
    
    Args:
        input_shape: tuple (timesteps, n_features)
        encoding_dim: int, size of the latent representation
        
    Returns:
        model: tf.keras.Model instance
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER ---
    
    # Multi-Scale CNN Branch 1 (Short-term patterns)
    conv1 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    pool1 = layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
    
    # Multi-Scale CNN Branch 2 (Medium-term patterns)
    conv2 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
    pool2 = layers.MaxPooling1D(pool_size=2, padding='same')(conv2)
    
    # Multi-Scale CNN Branch 3 (Long-term patterns)
    conv3 = layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(inputs)
    pool3 = layers.MaxPooling1D(pool_size=2, padding='same')(conv3)
    
    # Fusion
    merged = layers.concatenate([pool1, pool2, pool3], axis=-1)
    
    # Temporal Processing
    lstm_enc = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(merged)
    lstm_enc = layers.Dropout(0.2)(lstm_enc)
    
    # Attention (Optional, helps focus on key anomalies)
    # We use a simple dense reduction or flattening for latent if we don't use full attention
    # Let's use a Flatten + Dense for the bottleneck to force compression
    flatten = layers.Flatten()(lstm_enc)
    latent = layers.Dense(encoding_dim, activation='relu', name='latent_code')(flatten)
    
    # --- DECODER ---
    
    # Expand back to temporal structure
    # Note: We need to match the dimension after pooling. 
    # If original steps=20, pool(2) -> 10 steps.
    target_steps = input_shape[0] // 2 
    
    repeated = layers.RepeatVector(target_steps)(latent)
    
    lstm_dec = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(repeated)
    lstm_dec = layers.Dropout(0.2)(lstm_dec)
    
    # Upsampling to restore original length
    upsampled = layers.UpSampling1D(size=2)(lstm_dec)
    
    # Reconstruction Conv
    # Using Conv1D with same padding to refine features
    decoded = layers.Conv1D(filters=input_shape[1], kernel_size=3, activation='linear', padding='same')(upsampled)
    
    # Ensure output shape matches input shape exactly
    # Sometimes UpSampling might exceed original length if odd, need cropping logic if strict
    # For now assuming even input length (e.g., 20)
    
    model = models.Model(inputs=inputs, outputs=decoded, name="MS_CNN_BiLSTM_AE")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # Test instantiation
    model = build_multiscale_cnn_bilstm_ae((20, 78))
    model.summary()
