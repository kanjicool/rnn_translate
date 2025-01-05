from flask import Blueprint, render_template, redirect, url_for, request, jsonify
from flask_login import current_user
from rnn_translate.models import (
    predict_sequence,
    text_to_sequence,
    get_vocab,
    load_translation_model,
    translate_text,
)
import logging
import numpy as np
import sentencepiece as spm


module = Blueprint("site", __name__)


# โหลดโมเดลและ tokenizer
model = load_translation_model()
en_sp = spm.SentencePieceProcessor()
th_sp = spm.SentencePieceProcessor()
en_sp.load("rnn_translate/utils/data/en_sp.model")
th_sp.load("rnn_translate/utils/data/th_sp.model")

# # แปลงข้อความเป็น tokens (ตัวเลข) สำหรับภาษาอังกฤษ
# tokens_en = en_sp.encode("this is a test sentence", out_type=int)
# print(f"English Tokens: {tokens_en}")

# # แปลงข้อความเป็น tokens (ตัวเลข) สำหรับภาษาไทย
# tokens_th = th_sp.encode("นี่คือประโยคทดสอบ", out_type=int)
# print(f"Thai Tokens: {tokens_th}")

# # แปลงกลับเป็นข้อความจาก tokens ภาษาอังกฤษ
# text_en = en_sp.decode(tokens_en)
# print(f"Decoded English: {text_en}")

# # แปลงกลับเป็นข้อความจาก tokens ภาษาไทย
# text_th = th_sp.decode(tokens_th)
# print(f"Decoded Thai: {text_th}")

logging.basicConfig(level=logging.DEBUG)


@module.route("/")
def index():
    return render_template("/site/index.html")


@module.route("/translate", methods=["POST"])
def translate():
    # อ่านค่าจากฟอร์ม
    source_text = request.form.get("source_text")

    # ตรวจสอบข้อความที่รับมา
    print(f"Source Text: {source_text}")

    # แปลข้อความ
    try:
        print("------------------")
        print("...Start translate...")
        translated_text = translate_text(source_text, model, en_sp, th_sp)
        print(f"translated_text: {translated_text}")
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

    # ส่งผลลัพธ์กลับไปยังหน้าเว็บ
    return jsonify({"translated_sequence": f"{translated_text}"})


# @module.route("/translate", methods=["POST"])
# def translate():
#     model = load_translation_model()
#     print(f"Model : {model}")

#     # อ่านค่าจากฟอร์ม
#     source_language = request.form.get("source_language")
#     target_language = request.form.get("target_language")
#     source_text = request.form.get("source_text")

#     # ตรวจสอบข้อความที่รับมา
#     print(f"Source Language: {source_language}")
#     print(f"Target Language: {target_language}")
#     print(f"Source Text: {source_text}")

#     # ตรวจสอบข้อความที่รับมา
#     word_to_idx, idx_to_word, vocab_size = get_vocab()
#     input_sequence = text_to_sequence(source_text, word_to_idx)

#     print("------------------")
#     print(f"Text: {source_text}")
#     print(f"Index Sequence: {input_sequence}")

#     # สร้าง initial state (hidden state และ cell state ของ LSTM)
#     initial_state = [
#         np.zeros((1, 512)),  # hidden state
#         np.zeros((1, 512)),  # cell state
#     ]

#     # print(f"initial_state: {initial_state}")

#     # Translate Text!
#     try:
#         translated_sequence = predict_sequence(
#             input_sequence, model, vocab_size, idx_to_word, initial_state
#         )
#         translated_text = " ".join(translated_sequence)
#     except Exception as e:
#         print(f"Error during prediction: {str(e)}")
#         return jsonify({"error": "Failed to translate text"}), 500

#     # ส่งผลลัพธ์กลับไปยังหน้าเว็บ
#     return jsonify(
#         {
#             "translated_sequence": f"Translated ({source_language} -> {target_language}): {translated_text}"
#         }
#     )
