import mongoengine as me
from flask_mongoengine import MongoEngine
from .rnn_model import (
    get_vocab,
    load_translation_model,
    text_to_sequence,
    predict_sequence,
    translate_text,
)

__all__ = [
    "get_vocab",
    "load_translation_model",
    "text_to_sequence",
    "predict_sequence",
    "translate_text",
]

db = MongoEngine()


def init_db(app):
    db.init_app(app)
