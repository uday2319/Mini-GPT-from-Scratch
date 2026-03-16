from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer=Tokenizer(BPE())
tokenizer.pre_tokenizer=Whitespace()
trainer=BpeTrainer(
    vocab_size=5000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

)
tokenizer.train(["D:/Mini-GPT/data/datasetTinyShakesphere.txt"],trainer)
tokenizer.save("tokenizer/tokenizer.json")