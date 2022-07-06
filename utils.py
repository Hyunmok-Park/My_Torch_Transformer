import torch
#import sentencepiece

def make_input_tensor(input):
    # vocab loading
    vocab_file = "<path of data>/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    # 입력 texts
    lines = [
        "겨울은 추워요.",
        "감기 조심하세요."
    ]

    # text를 tensor로 변환
    inputs = []
    for line in lines:
        pieces = vocab.encode_as_pieces(line)
        ids = vocab.encode_as_ids(line)
        inputs.append(torch.tensor(ids))
        print(pieces)

    # 입력 길이가 다르므로 입력 최대 길이에 맟춰 padding(0)을 추가 해 줌
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs

def make_attn_mask(inputs, input_sum):
    attn_mask = inputs.eq(0).unsqueeze(1).expand(input_sum.size(0), input_sum.size(1), input_sum.size(1))
    return attn_mask

