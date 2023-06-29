import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

def contains_unwanted_words(text, unwanted_words):
    """
    Check if the input text contains any unwanted words or phrases.
    
    Args:
        text (str): The input text.
        unwanted_words (list): The list of unwanted words or phrases.
        
    Returns:
        bool: True if any unwanted word or phrase is found in the input text, False otherwise.
    """

    for phrase in unwanted_words:
        if phrase.lower() in text.lower():
            return True
    return False

def main(args):
    """
    Main function to load the dataset, filter out the unwanted conversations,
    and save the new dataset and the removed entries into separate files.
    
    Args:
        args: Command line arguments.
    """

    in_file = args.in_file
    out_file = args.out_file
    removed_file = args.removed_file if args.removed_file else out_file.split(".")[0] + "_removed.json"

    unwanted_words = [
		"text-based AI language model",
		"please refrain",
		"responsible AI",
		"AI principles",
		"AI assistant",
		"an AI language",
		"as a language model",
		"as an AI language model",
		"As a large language model",
		"As an AI",
		"it is not appropriate",
		"it's not appropriate",
		"I cannot fulfill your request",
		"my guidelines",
		"prioritize user safety",
		"well-being of all users",
		"cannot provide guidance",
		"cannot provide information",
		"unable to offer assistance",
		"cannot engage in discussions",
		"programming prohibits",
		"against my programming",
		"designed to prioritize safety",
		"not able to provide",
		"maintain user safety",
		"adhere to safety guidelines",
		"cannot provide any information",
		"an AI language model you don't have",
		"As an AI language model, I cannot",
		"As an AI language model, I do not",
		"As an AI language model, I am not able",
		"As an AI language model, I don't have personal",
		"I am an AI language model and do not",
		"As an AI language model, I don't have",
		"As an AI language model, I am only able",
		"AI language model and I do not",
		"As an AI language model, I cannot modify",
		"As an AI language model, I do not",
		"I know as an AI language model you don't have",
		"as an AI language model, you cannot",
		"I'm sorry, but as an AI language model",
		"As an AI language model, I don't have",
		"Unfortunately, I cannot provide",
		"I'm sorry, I cannot",
		"I'm sorry, I cannot generate",
		"AI cannot create or program",
		"I'm afraid I cannot create",
		"you cannot create an",
		"legal and ethical",
		"engage in unethical",
		"como modelo de lenguaje AI",
		"Lo siento, como modelo de lenguaje",
		"no puedo proporcionar",
		"pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
		"Lo siento, pero no puedo",
		"Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
		"Lo siento, como modelo de lenguaje, no tengo",
		"Lo siento, debe haber habido una confusi\u00f3n",
		"Lo siento, como modelo de lenguaje, no puedo realizar",
		"Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
		"Lamento no poder proporcionarte el c\u00f3digo",
		"Desculpe-me, mas a linguagem vulgar e ofensiva",
		"apropriada em nenhum contexto",
		"Como modelo de linguagem",
		"Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
		"I cannot assist",
		"I'm an AI" ,
		"I am an AI",
		"not a human",
		"I am an AI",
		"a language model",
		"As a machine",
		"I don't have the ability",
		"I am here to assist",
		"my purpose is to ",
		"my knowledge cutoff",
		"my knowledge cut off",
		"September 2021",
		"I apologize, but",
		"my programming",
		"*This chat conversation is shared from",
		"*This conversation is shared from"
    ]

    from datasets import DatasetDict
    dataset = DatasetDict.from_parquet({'train': in_file}) 
    # Load the data into a Hugging Face dataset
    if ".json" in in_file:
        dataset = load_dataset("json", data_files=in_file)
    elif ".parquet" in in_file:
        dataset = load_dataset("parquet", data_files=in_file)

    #dataset = load_from_disk()

    # Use the filter method to separate the entries
    new_dataset = dataset.filter(lambda x: not contains_unwanted_words(x['response'], unwanted_words))
    removed_dataset = dataset.filter(lambda x: contains_unwanted_words(x['response'], unwanted_words))

    print(f"Returned {len(new_dataset)} out of {len(dataset)}, start dump ...")
    new_dataset.save_to_disk(out_file)

    print(f"Removed {len(removed_dataset)} entries, dumping to removed file ...")
    removed_dataset.save_to_disk(removed_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Removes most OpenAI disclaimers, refusals, etc from a dataset\'s output field.')
    parser.add_argument('--in_file', type=str, help='Input file', required=True)
    parser.add_argument('--out_file', type=str, help='Output file', required=True)
    parser.add_argument('--removed_file', type=str, help='Removed entries file', default=None)
    args = parser.parse_args()

    main(args)
