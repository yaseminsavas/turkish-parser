def get_embedding(embedding_directory):

    external_embedding = {}
    with open(embedding_directory, 'r') as file:
        for index, line in enumerate(file):
            key = line.split(' ')[0]
            value = [float(f) for f in line.strip().split(' ')[1:]]
            external_embedding[key] = value

    print("External embeddings are gathered. Vector dimension: ", len(external_embedding[key]))

    return external_embedding