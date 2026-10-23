"""Export facebook/m2m100_418M (MIT) to ONNX int8 for GrabNConvert's subtitle translation.

Writes into OUT: encoder.onnx, decoder.onnx (first step), decoder_past.onnx (steps with the cache),
all dynamic int8 with every weight that two of them share stored once in weights.bin, tokenizer.json
(a Unigram tokenizer in the model's own id order), languages.json, and prints the sessions'
input/output names (the Rust side matches by name).
"""
import json, os, sys, shutil
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "m2m100-onnx")
MODEL = "facebook/m2m100_418M"
OUT.mkdir(parents=True, exist_ok=True)
work = OUT / "fp32"

from optimum.exporters.onnx import main_export
if not (work / "decoder_with_past_model.onnx").exists():
    main_export(MODEL, output=work, task="text2text-generation-with-past", opset=17, no_post_process=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx, hashlib
from onnx.external_data_helper import set_external_data
q = OUT / "q"; q.mkdir(exist_ok=True)
names = [("encoder_model.onnx", "encoder.onnx"), ("decoder_model.onnx", "decoder.onnx"), ("decoder_with_past_model.onnx", "decoder_past.onnx")]
for src, dst in names:
    if not (q / dst).exists():
        quantize_dynamic(str(work / src), str(q / dst), weight_type=QuantType.QInt8, per_channel=False, reduce_range=False)
# The embedding table and the output layer are the same bytes in every graph: store each blob once.
if not (OUT / "weights.bin").exists():
    seen = {}
    with open(OUT / "weights.bin", "wb") as blob:
        for _, dst in names:
            m = onnx.load(str(q / dst))
            for t in m.graph.initializer:
                if len(t.raw_data) < (1 << 20):
                    continue
                h = hashlib.sha256(t.raw_data).hexdigest()
                if h not in seen:
                    seen[h] = (blob.tell(), len(t.raw_data))
                    blob.write(t.raw_data)
                off, ln = seen[h]
                set_external_data(t, "weights.bin", offset=off, length=ln)
                t.ClearField("raw_data")
            onnx.save(m, str(OUT / dst))
for _, dst in names:
    print(dst, round((OUT / dst).stat().st_size / 1e6, 1), "MB")
print("weights.bin", round((OUT / "weights.bin").stat().st_size / 1e6, 1), "MB")

# Tokenizer: M2M100's SentencePiece model is a BPE model (its piece scores are merge ranks, not
# log-probabilities), so build a BPE tokenizer the way transformers' slow→fast converter does, then
# renumber it in the model's own id order (vocab.json, then the language tokens).
from transformers import M2M100Tokenizer
from transformers.convert_slow_tokenizer import SentencePieceExtractor
tok = M2M100Tokenizer.from_pretrained(MODEL)
sp_vocab, merges = SentencePieceExtractor(tok.spm_file).extract()
vocab = json.load(open(tok.vocab_file))  # token -> id (model ids)
added = json.load(open(work / "added_tokens.json"))  # language tokens, ids after the vocab
assert sorted(added.values()) == list(range(len(vocab), len(vocab) + len(added))), "added token ids are not dense"
import tokenizers
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, AddedToken
missing = [p for p in sp_vocab if p not in vocab]
print("pieces not in the model's vocab:", len(missing), missing[:5])
bpe_vocab = {t: i for t, i in vocab.items()}
merges = [(a, b) for a, b in merges if a in bpe_vocab and b in bpe_vocab and a + b in bpe_vocab]
tk = Tokenizer(models.BPE(vocab=bpe_vocab, merges=merges, unk_token="<unk>", fuse_unk=True, byte_fallback=False))
tk.normalizer = normalizers.Sequence([normalizers.Nmt(), normalizers.NFKC(), normalizers.Replace(tokenizers.Regex(r" {2,}"), " ")])
tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
for t, _ in sorted(added.items(), key=lambda kv: kv[1]):
    tk.add_special_tokens([AddedToken(t, special=True)])
tk.add_special_tokens([AddedToken(t, special=True) for t in ["<s>", "<pad>", "</s>", "<unk>"]])
for t in ["<s>", "<pad>", "</s>", "<unk>"] + list(added):
    assert tk.token_to_id(t) == vocab.get(t, added.get(t)), (t, tk.token_to_id(t))
vocab.update(added)
langs = sorted(t.strip("_") for t in added)
# The ids must match the reference tokenizer exactly, on several scripts.
for src, text in [("en", "The meeting moved to Thursday at 5 pm."), ("en", "Good morning everyone, thanks for coming!"), ("bn", "আগামীকাল সকালে অফিসে আসবেন।"), ("hi", "निकटतम रेलवे स्टेशन कहाँ है?"), ("ja", "明日の朝、オフィスに来てください。"), ("de", "Schönen Tag noch – bis später!"), ("en", "Email me at a.b@example.com, it's 10:30am.")]:
    tok.src_lang = src
    ref = tok(text).input_ids
    mine = [vocab[f"__{src}__"]] + tk.encode(text).ids + [vocab["</s>"]]
    assert ref == mine, (text, tok.convert_ids_to_tokens(ref), [tk.id_to_token(i) for i in mine])
print("tokenizer matches the reference")
tk.save(str(OUT / "tokenizer.json"))
json.dump({"languages": langs, "eos": vocab["</s>"], "pad": vocab["<pad>"], "bos": vocab["<s>"], "unk": vocab["<unk>"]}, open(OUT / "languages.json", "w"), indent=1)

# A real check through ONNX Runtime, and the names the Rust side needs.
import onnxruntime as ort, numpy as np
for name in ["encoder.onnx", "decoder.onnx", "decoder_past.onnx"]:
    s_ = ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])
    print(name, "inputs:", [(i.name, i.shape) for i in s_.get_inputs()][:6], "... outputs:", [o.name for o in s_.get_outputs()][:4])

enc = ort.InferenceSession(str(OUT / "encoder.onnx"), providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(str(OUT / "decoder.onnx"), providers=["CPUExecutionProvider"])
decp = ort.InferenceSession(str(OUT / "decoder_past.onnx"), providers=["CPUExecutionProvider"])
past_names = [i.name for i in decp.get_inputs() if i.name.startswith("past_key_values")]
wants_hidden = any(i.name == "encoder_hidden_states" for i in decp.get_inputs())
def translate(text, src, tgt):
    ids = [vocab[f"__{src}__"]] + tk.encode(text).ids + [vocab["</s>"]]
    input_ids = np.array([ids], dtype=np.int64); mask = np.ones_like(input_ids)
    hidden = enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]
    out_ids = [vocab["</s>"], vocab[f"__{tgt}__"]]
    outs = dec.run(None, {"input_ids": np.array([out_ids], dtype=np.int64), "encoder_hidden_states": hidden, "encoder_attention_mask": mask})
    names = [o.name for o in dec.get_outputs()]
    past = {n.replace("present", "past_key_values"): v for n, v in zip(names[1:], outs[1:])}
    logits = outs[0]
    for _ in range(80):
        nxt = int(np.argmax(logits[0, -1]))
        out_ids.append(nxt)
        if nxt == vocab["</s>"]:
            break
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64), "encoder_attention_mask": mask}
        if wants_hidden:
            feed["encoder_hidden_states"] = hidden
        for n in past_names:
            feed[n] = past[n]
        o = decp.run(None, feed)
        pn = [x.name for x in decp.get_outputs()]
        logits = o[0]
        for n, v in zip(pn[1:], o[1:]):
            past[n.replace("present", "past_key_values")] = v
    return tk.decode([i for i in out_ids[2:] if i != vocab["</s>"]], skip_special_tokens=True)
print("en->bn:", translate("The meeting moved to Thursday at 5 pm. Please bring the contract.", "en", "bn"))
print("bn->en:", translate("আমি আগামীকাল ঢাকায় যাব।", "bn", "en"))
print("en->hi:", translate("Where is the nearest train station?", "en", "hi"))
