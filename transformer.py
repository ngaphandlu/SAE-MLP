import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras import ops
import numpy as np

class RMSNormalization(layers.Layer):
    """RMS Normalization layer - you'll need to implement this if not available"""
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        mean_square = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x_normalized = x / tf.sqrt(mean_square + self.epsilon)
        return self.scale * x_normalized
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

class AttentionPooling(layers.Layer):
    """Attention-based pooling layer"""
    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, embed_dim)
        attention_scores = tf.matmul(inputs, self.attention_weights)  # (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        weighted_output = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch_size, embed_dim)
        return weighted_output
    
    def get_config(self):
        return super().get_config()

class GLU(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return a * tf.sigmoid(b)

    def get_config(self):
        config = super(GLU, self).get_config()
        config.update({
            'axis': self.axis,
        })
        return config
class SwiGLU(layers.Layer):
    """SwiGLU activation function used in LLaMA and PaLM"""
    def __init__(self, axis=-1, **kwargs):
        super(SwiGLU, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return a * tf.nn.swish(b)  # Swish activation on gate
    
    def get_config(self):
        config = super(SwiGLU, self).get_config()
        config.update({'axis': self.axis})
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            #layers.Dense(ff_dim*2), GLU(),layers.Dense(embed_dim)
            #layers.Dense(ff_dim * 2, use_bias=False), SwiGLU(), layers.Dense(embed_dim, use_bias=False)
            layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim)
            ])
        self.layernorm1 = RMSNormalization(epsilon=1e-6) 
        self.layernorm2 = RMSNormalization(epsilon=1e-6) 
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
    
    def call(self, inputs, training=False):
        # # Post-norm architecture
        # attn_output = self.att(inputs, inputs, inputs)
        # attn_output = self.dropout1(attn_output, training=training)
        # out1 = self.layernorm1(inputs + attn_output)
        # ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        # return self.layernorm2(out1 + ffn_output)
        
        # Pre-norm architecture (RMSNorm before attention)
        attn_input = self.layernorm1(inputs)
        attn_output = self.att(attn_input, attn_input, attn_input)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output  # Residual connection
    
        ffn_input = self.layernorm2(out1)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output  # Residual connection
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

class TokenandPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, input_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.input_dim = input_dim  # Input feature dimension (1 in your case)
        self.embed_dim = embed_dim  # Output embedding dimension
        # For continuous features, use Dense instead of Embedding
        self.feature_projection = layers.Dense(embed_dim, activation='linear')
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Create position indices
        positions = tf.range(start=0, limit=seq_len, delta=1)
        # Get positional embeddings
        position_embeddings = self.pos_emb(positions)
        # Project input features to embedding dimension
        feature_embeddings = self.feature_projection(x)
        # Add position and feature embeddings
        return feature_embeddings + position_embeddings
        
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'input_dim': self.input_dim,
            'embed_dim': self.embed_dim
        })
        return config

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.trans_blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=rate)
            for _ in range(num_layers)  
        ]
        # Final layer norm
        self.final_layernorm = RMSNormalization(epsilon=1e-6) #**** 27/06
    
    def call(self, x, training=False):
        for i in range(self.num_layers):
            x = self.trans_blocks[i](x, training=training)
        #return x
        return self.final_layernorm(x)  #**** 27/06
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):  # Fixed: double underscores
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_len, d_model)
   
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
        })
        return config
   
    def positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
   
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
   
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class WiFiRSSEmbedding(layers.Layer):
    """Custom embedding layer for WiFi RSS values with positional encoding"""
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim

        # For RSS values, use Dense projection instead of token embedding
        #self.rss_projection = layers.Dense(embed_dim, activation='gelu')
        # Use SwiGLU for RSS projection as well
        self.rss_projection = Sequential([
            layers.Dense(embed_dim * 2, use_bias=False),
            SwiGLU(),
            layers.Dense(embed_dim, use_bias=False)
        ])

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        #self.pos_encoding = PositionalEncoding(maxlen, embed_dim)  # Use sinusoidal

        self.layernorm = RMSNormalization()
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Create position indices for each access point
        positions = tf.range(start=0, limit=seq_len, delta=1)
        # Get positional embeddings
        position_embeddings = self.pos_emb(positions)
        # Project RSS features to embedding dimension
        rss_embeddings = self.rss_projection(x)
        # Combine RSS and positional information
        embeddings = rss_embeddings + position_embeddings
        return self.layernorm(embeddings)

        """ # Project RSS features to embedding dimension
        rss_embeddings = self.rss_projection(x)
        # Add positional encoding
        embeddings = self.pos_encoding(rss_embeddings)
        return self.layernorm(embeddings) """
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim
        })
        return config

