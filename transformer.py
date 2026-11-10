import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import ops
import numpy as np

register = keras.utils.register_keras_serializable(package="wifi")

@register
class RMSNormalization(layers.Layer):
    """RMS Normalization"""
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # Tính trong float32 để ổn định khi dùng mixed precision
        x32 = tf.cast(x, tf.float32)
        mean_square = tf.reduce_mean(tf.square(x32), axis=-1, keepdims=True)
        x_norm = x32 * tf.math.rsqrt(mean_square + self.epsilon)
        return tf.cast(x_norm, x.dtype) * tf.cast(self.scale, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@register
class AttentionPooling(layers.Layer):
    """Attention-based pooling: (batch, seq, dim) -> (batch, dim)"""
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.matmul(inputs, self.attention_weights)      # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)          # (B, D)

    def get_config(self):
        return super().get_config()


@register
class GLU(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return a * tf.sigmoid(b)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@register
class SwiGLU(layers.Layer):
    """SwiGLU (LLaMA / PaLM): swish(gate) * value"""
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return a * tf.nn.swish(b)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@register
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) phải chia hết cho num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # key_dim là kích thước MỖI head -> embed_dim // num_heads
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim #// num_heads
        )
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = RMSNormalization(epsilon=1e-6)
        self.layernorm2 = RMSNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        # TransformerBlock
    def build(self, input_shape):          # (batch, seq, embed_dim)
        self.layernorm1.build(input_shape)
        self.att.build(input_shape, input_shape)
        self.layernorm2.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Pre-norm
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output

        y = self.layernorm2(out1)
        ffn_output = self.ffn(y, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


@register
class TokenandPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, input_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.feature_projection = layers.Dense(embed_dim, activation="linear")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        position_embeddings = self.pos_emb(positions)           # (T, D)
        feature_embeddings = self.feature_projection(x)         # (B, T, D)
        return feature_embeddings + position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "input_dim": self.input_dim,
            "embed_dim": self.embed_dim,
        })
        return config


@register
class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.trans_blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                             ff_dim=ff_dim, rate=rate)
            for _ in range(num_layers)
        ]
        self.final_layernorm = RMSNormalization(epsilon=1e-6)

        # TransformerEncoder
    def build(self, input_shape):
        for block in self.trans_blocks:
            block.build(input_shape)
        self.final_layernorm.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        for block in self.trans_blocks:
            x = block(x, training=training)
        return self.final_layernorm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


@register
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding"""
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        if d_model % 2 != 0:
            raise ValueError("d_model phải là số chẵn cho sinusoidal encoding")
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(max_len, d_model)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, max_len, d_model):
        angle_rads = self._get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pe = tf.cast(self.pos_encoding[:, :seq_len, :], inputs.dtype)
        return inputs + pe

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


@register
class WiFiRSSEmbedding(layers.Layer):
    """Embedding cho giá trị RSS WiFi (mỗi AP là một token) + positional embedding"""
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.rss_projection = Sequential([
            layers.Dense(embed_dim * 2, use_bias=False),
            SwiGLU(),
            layers.Dense(embed_dim, use_bias=False),
        ])
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.layernorm = RMSNormalization()

        # WiFiRSSEmbedding
    def build(self, input_shape):          # input_shape = (batch, num_APs, 1)
        self.rss_projection.build(input_shape)
        self.pos_emb.build((None,))
        self.layernorm.build(tuple(input_shape[:-1]) + (self.embed_dim,))
        super().build(input_shape)

    def call(self, x):
        # x: (batch, num_APs, 1)
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        position_embeddings = self.pos_emb(positions)           # (T, D)
        rss_embeddings = self.rss_projection(x)                 # (B, T, D)
        return self.layernorm(rss_embeddings + position_embeddings)

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config
