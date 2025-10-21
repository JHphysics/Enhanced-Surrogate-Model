import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# =========================
# This code uses MNIST. However, since MNIST labels are one-hot encoded, the number of possible latent vectors is limited to 10. 
# Please note that this is just for demonstration purposes.
# Just so you know, I usually write really messy code, so I left the cleanup to ChatGPT.
# ========================


# =========================
# 1. Prepare data
# =========================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten images
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

num_classes = 10
# Convert labels to one-hot encoding
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# =========================
# 2. Define autoencoder (multi-output)
# =========================
input_dim = 28*28
latent_dim = num_classes
hidden_units = 128

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(hidden_units, activation='relu')(inputs)
latent = Dense(latent_dim, activation='softmax', name='latent')(h)

# Decoder
h_dec = Dense(hidden_units, activation='relu')(latent)
outputs = Dense(input_dim, activation='sigmoid', name='reconstruction')(h_dec)

# Multi-output model: reconstruction + latent
autoencoder = Model(inputs, [outputs, latent])

# =========================
# 3. Compile model
# =========================
losses = {'reconstruction': 'mse', 'latent': 'mse'}
loss_weights = {'reconstruction': 1.0, 'latent': 1.0}

autoencoder.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

# =========================
# 4. Train the model
# =========================
autoencoder.fit(
    x_train_flat,
    {'reconstruction': x_train_flat, 'latent': y_train_onehot},
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_flat, {'reconstruction': x_test_flat, 'latent': y_test_onehot})
)

# =========================
# 5. Extract latent vectors
# =========================
encoder = Model(inputs, autoencoder.get_layer('latent').output)
latent_test = encoder.predict(x_test_flat)

# =========================
# 6. Visualize latent vectors (average per class)
# =========================
avg_latent = np.zeros((num_classes, latent_dim))
for i in range(num_classes):
    avg_latent[i] = np.mean(latent_test[y_test == i], axis=0)

plt.figure(figsize=(10, 6))
plt.imshow(avg_latent, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Latent dimension')
plt.ylabel('Digit class')
plt.title('Average latent vectors for each digit class')
plt.show()

# =========================
# 7. Create decoder model (latent -> reconstruction)
# =========================
latent_input = Input(shape=(latent_dim,))
h_dec_input = autoencoder.layers[-2](latent_input)
decoder_output = autoencoder.layers[-1](h_dec_input)
decoder = Model(latent_input, decoder_output)

# =========================
# 8. Generate images from one-hot latent vectors
# =========================
onehot_inputs = np.eye(num_classes)
generated_imgs = decoder.predict(onehot_inputs)

plt.figure(figsize=(12, 2))
for i in range(num_classes):
    plt.subplot(1, num_classes, i+1)
    plt.imshow(generated_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(str(i))
plt.suptitle('Decoder outputs from one-hot latent vectors')
plt.show()

# =========================
# 9. Compare original images with reconstructed images (random samples)
# =========================
num_samples = 10
plt.figure(figsize=(12, 4))
for i in range(num_samples):
    # Original image
    plt.subplot(2, num_samples, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel('Original')
    
    # Reconstructed image
    reconstructed = autoencoder.predict(x_test_flat[i:i+1])[0]
    plt.subplot(2, num_samples, i+1+num_samples)
    plt.imshow(reconstructed.reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel('Reconstructed')
plt.show()
