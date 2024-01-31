from transformers import RobertaTokenizer, RobertaModel

def download():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.save_pretrained("./tokenizer_roberta")

    model = RobertaModel.from_pretrained("roberta-base")
    model.save_pretrained("./model_roberta")

if __name__ == '__main__':
    download()
    