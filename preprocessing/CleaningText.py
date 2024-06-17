import re

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text):
    # Remove múltiplos espaços e quebras de linha
    text = re.sub(r'\s+', ' ', text)
    
    # Remove caracteres não-ASCII (exceto alguns comuns como acentos)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Substitui pontuações incomuns e caracteres especiais por espaço
    text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚãõÃÕâêîôûÂÊÎÔÛçÇ.,;?!\'"()\- ]', ' ', text)
    
    # Remove múltiplos espaços novamente (após substituições)
    text = re.sub(r'\s+', ' ', text)

    # Remove espaços no início e fim do texto
    text = text.strip()
    
    return text

def save_text(cleaned_text, save_path):
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

def process_text_file(input_path, output_path):
    text = load_text(input_path)
    cleaned_text = clean_text(text)
    save_text(cleaned_text, output_path)

if __name__ == "__main__":
    input_path = 'C:/Users/SuperBusiness.DESKTOP-V6R5K91/Desktop/CLOD/datasets/booksprocessed.txt'  # Caminho do arquivo de entrada
    output_path = 'C:/Users/SuperBusiness.DESKTOP-V6R5K91/Desktop/CLOD/datasets/textcleaned.txt'  # Caminho do arquivo de saída
    process_text_file(input_path, output_path)