from config import *
from net import *
from utils import *
from pos_embedding import *

def main(conf):

    inputs = torch.tensor([
        [3091, 3604,  206, 3958, 3760, 3590,    0,    0],
        [ 212, 3605,   53, 3832, 3596, 3682, 3760, 3590]
    ])

    word_emb = my_word_emb(len_vocab=5000, hidden_dim=conf.hidden_dim)
    inputs_embs = word_emb(inputs)

    pos_encoding = get_sinusoid_encoding_table(conf.n_seq, conf.hidden_dim)
    pos_encoding = torch.FloatTensor(pos_encoding)

    pos_emb = my_pos_emb(pos_encoding)
    pos_embs = pos_emb(inputs)

    input_sum = inputs_embs + pos_embs

    encoder = my_enc(conf)
    attn_mask = make_attn_mask(inputs, input_sum)
    output = encoder(srcQ=input_sum, srcK=input_sum, srcV=input_sum, attn_mask=attn_mask)
    encoder2 = my_enc(conf)
    output2 = encoder2(srcQ=output, srcK=output, srcV=output, attn_mask=attn_mask)

    return 0

if __name__ == "__main__":
    conf, unparsed = get_args()
    main(conf)