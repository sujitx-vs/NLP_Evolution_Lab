from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(10000, 128)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(
    256,
    return_state=True
)(encoder_embedding)

encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(10000, 128)(decoder_inputs)

decoder_lstm = LSTM(
    256,
    return_sequences=True,
    return_state=True
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

decoder_outputs = Dense(10000, activation="softmax")(decoder_outputs)

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.summary()