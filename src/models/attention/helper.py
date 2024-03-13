"""
Helper functions for attention models.
"""


def get_dimensions_from_encoder_outputs(encoder_outputs):
    batch_size, seq_len, encoder_hidden_size = encoder_outputs.shape

    return batch_size, seq_len, encoder_hidden_size
