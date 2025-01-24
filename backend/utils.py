from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # 1000 chars
        chunk_overlap = 200,
        length_function = len # len fn from python
    )

    chunks = text_splitter.split_text(raw_text)
    # print(chunks)
    return chunks
