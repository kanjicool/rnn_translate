import os
import numpy as np
from tensorflow.keras.models import load_model
import sentencepiece as spm


# ฟังก์ชันสร้าง word_to_idx และ idx_to_word
# def get_vocab(vocab_file_path="../utils/rnn_model/vocab.txt"):
#     """
#     สร้างพจนานุกรม word-to-index และ index-to-word จากไฟล์
#     """
#     if not os.path.exists(vocab_file_path):
#         raise FileNotFoundError(f"Vocabulary file not found at {vocab_file_path}")

#     with open(vocab_file_path, "r", encoding="utf-8") as f:
#         vocab = [line.strip() for line in f]

#     word_to_idx = {word: idx for idx, word in enumerate(vocab)}
#     idx_to_word = {idx: word for idx, word in enumerate(vocab)}
#     vocab_size = len(vocab)
#     return word_to_idx, idx_to_word, vocab_size


# โหลดโมเดล
def load_translation_model():
    """
    โหลดโมเดล RNN
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        base_dir, "../utils/rnn_model/seq2seq_translation_model.h5"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model(model_path)
    print(f"\n... Complete loaded model ...")
    print(f"Model loaded from: {model_path}")
    return model


def get_vocab():
    """
    สร้าง dict word-to-index และ index-to-word จากไฟล์
    """
    vocab = ["<PAD>", "<UNK>", "hello", "world"]

    word_to_idx = {"hello": 1, "world": 2, "<UNK>": 3, "<PAD>": 0}
    idx_to_word = {4: "สวัสดี", 5: "โลก", 0: "<PAD>"}
    vocab_size = len(vocab)
    return word_to_idx, idx_to_word, vocab_size


# ฟังก์ชันแปลงข้อความเป็นลำดับดัชนี
def text_to_sequence(text, word_to_idx):
    """
    แปลงข้อความเป็นลำดับดัชนี
    """
    words = text.split()  # แยกคำด้วยช่องว่าง
    sequence = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
    return sequence


# ฟังก์ชันพยากรณ์ผลลัพธ์จากโมเดล
def predict_sequence(input_sequence, model, vocab_size, idx_to_word, initial_state):
    """
    แปลงลำดับข้อความ (sequence) เพื่อพยากรณ์ด้วยโมเดลที่โหลดไว้
    """
    # ปรับข้อมูล input sequence ให้มี batch dimension
    input_sequence = np.array(input_sequence).reshape((1, len(input_sequence)))

    # พยากรณ์ผลลัพธ์จากโมเดล
    # ส่ง input_sequence และ initial_state ให้กับโมเดล
    predictions = model.predict([input_sequence, initial_state])  # ส่งเป็น 2 inputs

    # แปลงผลลัพธ์กลับไปเป็นข้อความ
    predicted_sequence = [idx_to_word[np.argmax(pred)] for pred in predictions[0]]

    return predicted_sequence


# ฟังก์ชันแปลข้อความ
def translate_text(input_text, model, en_sp, th_sp, max_len=50):
    import numpy as np

    # ทำความสะอาดข้อความ
    text = input_text.lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())

    # แปลงเป็น tokens ด้วย English tokenizer
    tokens = en_sp.encode(text, out_type=int)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    print(f"tokens : {tokens}")

    # เพิ่ม padding
    input_sequence = np.array(
        [np.pad(tokens, (0, max_len - len(tokens)), "constant")], dtype=np.float32
    )
    decoder_input = np.zeros((1, max_len), dtype=np.float32)
    decoder_input[0, 0] = 2  # start token

    # ทำนายผลลัพธ์
    predicted_sequence = []
    for t in range(max_len):
        output = model.predict([input_sequence, decoder_input], verbose=0)
        next_token = int(np.argmax(output[0, t]))  # แปลงเป็น int
        print(
            f"Step {t}: Token {next_token}, Decoded: {th_sp.id_to_piece(next_token)}"
        )  # Debugging

        if next_token == th_sp.piece_to_id("<end>") or (
            len(predicted_sequence) > 1 and next_token == predicted_sequence[-1]
        ):
            break
        predicted_sequence.append(next_token)

        if t + 1 < max_len:
            decoder_input[0, t + 1] = next_token

    print(f"predicted_sequence: {predicted_sequence}")

    # กรอง token ที่ไม่สามารถถอดรหัสได้
    predicted_sequence = [
        token for token in predicted_sequence if token in range(len(th_sp))
    ]

    result = th_sp.decode(predicted_sequence) if predicted_sequence else ""

    return result
