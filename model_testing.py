from ru_word2number import w2n
import os
import torch
import torchaudio
import faiss
import numpy as np
import json
import pickle
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import time
count = 0
from transformers.data import processors
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
# model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
# model_name = "emre/wav2vec2-xls-r-300m-Russian-small"
model_name = 'bond005/wav2vec2-base-ru-birm'
model = Wav2Vec2Model.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)
text_model = Wav2Vec2ForCTC.from_pretrained(model_name)
# Function to load audio
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


# Function to get embeddings
def get_wav2vec_embedding(waveform, sample_rate):
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state  # Shape: (1, seq_len, hidden_size)
    embeddings = embeddings.mean(dim=1)  # Shape: (1, hidden_size)
    return embeddings  # Shape: (1, hidden_size)

def retrive_text(waveform, sample_rate):
    waveform = waveform.squeeze()
    input_values = processor(waveform, sampling=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = text_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]
# Function to index audio files and their labels
# Modified function to index audio files and metadata
def index_audio_files(audio_dir, json_data_path, faiss_index_path, labels_path, metadata_path):
    global count

    # Load data from JSON
    with open(json_data_path, 'r') as f:
        json_data = json.load(f)

    # Check JSON format
    if not isinstance(json_data, list):
        raise ValueError("JSON data should be a list of dictionaries")

    # Collect all audio file paths
    audio_paths = [js['audio_filepath'] for js in json_data]

    # Initialize variables for FAISS, labels, and metadata
    labels = []
    metadata = []
    index = None
    # Process each audio file
    for audio_path in audio_paths:
        full_audio_path = os.path.join(audio_dir, audio_path)
        print(f"Processing file: {full_audio_path}")

        waveform, sample_rate = load_audio(full_audio_path)
        embedding = get_wav2vec_embedding(waveform, sample_rate)
        embeddings = embedding.numpy()  # Shape: (1, hidden_size)

        count += 1
        print(f"Embeddings processed: {count}")

        if index is None:
            vector_dim = embeddings.shape[1]  # embeddings.shape = (1, hidden_size)
            index = faiss.IndexFlatL2(vector_dim)

        # Add embeddings to FAISS
        index.add(embeddings)  # embeddings is (1, hidden_size)

        # Find label and metadata for this audio file in JSON and add it
        label_found = False
        for js in json_data:
            if audio_path == js['audio_filepath']:
                labels.append(js['label'])
                metadata.append({
                    "text": js.get('text', ''),
                    "attribute": js.get('attribute', ''),
                    "id": js.get('id', ''),
                    "label":js.get('label','')
                })
                label_found = True
                break

        if not label_found:
            print(f"Label not found for: {audio_path}")

    # Save FAISS index
    faiss.write_index(index, faiss_index_path)

    # Save labels
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)

    # Save metadata
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index, labels, and metadata successfully saved to {faiss_index_path}, {labels_path}, and {metadata_path}")
# Function to find nearest neighbor
# Modified function to find nearest neighbor and retrieve metadata
# Function to find nearest neighbor
def find_nearest_neighbor(audio_path, faiss_index_path, labels_path, metadata_path, config):
    # Load FAISS index, labels, and metadata
    index = faiss.read_index(faiss_index_path)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Load and process audio file
    waveform, sample_rate = load_audio(audio_path)
    embedding = get_wav2vec_embedding(waveform, sample_rate)
    embedding_np = embedding.numpy()  # Shape: (1, hidden_size)

    # Search for nearest neighbor using the configuration
    distances, indices = search_with_faiss(index, embedding_np, config)

    # Get nearest neighbor's index, label, and metadata
    nearest_index = indices[0][0]
    nearest_label = labels[nearest_index]
    nearest_metadata = metadata[nearest_index]

    return nearest_label, nearest_metadata, distances[0][0]

def show_first_n_vectors_with_metadata(faiss_index_path, metadata_path, n=4):
    # Load FAISS index and metadata
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Check if there are enough vectors to display
    total_vectors = index.ntotal
    if total_vectors == 0:
        print("No vectors found in the index.")
        return
    
    # Ensure N does not exceed total vectors
    n = min(n, total_vectors)

    # Show the first N vectors and their metadata
    print(f"Showing the first {n} vectors with metadata:")
    for i in range(n):
        print(f"\nVector {i+1}:")
        print(f"Metadata: {metadata[i]}")

# Function to compare metadata between test and train
# Function to compare labels between test and train
def compare_label(test_label, train_label):
    # Return 1 if labels match, else 0
    return 1 if test_label == train_label else 0
# Function to search FAISS index with dynamic configuration
def search_with_faiss(index, query_embedding, config):
    # Optional normalization for cosine similarity (using inner product)
    if config.get('normalize', False):
        faiss.normalize_L2(query_embedding)
    
    k = config.get('k', 1)  # Number of nearest neighbors to retrieve, default to 5
    
    # Perform the search on the FAISS index
    distances, indices = index.search(query_embedding, k)
    
    return distances, indices


def extract_arg(text):

    split_text = text.split(' ')

    _sum = 0

    for word in split_text:
        try:
            retriw = w2n.word_to_num(word)
            _sum += retriw
        except ValueError:
            continue

    return _sum


# Function to find nearest neighbor for all test samples and compare labels
def evaluate_test_set(test_audio_dir, test_json_data, faiss_index_path, labels_path, metadata_path, config, output_json_path):
    # Load test data from JSON
    global processor
    with open(test_json_data, 'r') as f:
        test_json = json.load(f)

    # Load FAISS index, labels, and metadata from training data
    index = faiss.read_index(faiss_index_path)
    with open(labels_path, "rb") as f:
        train_labels = pickle.load(f)
    with open(metadata_path, "rb") as f:
        train_metadata = pickle.load(f)

    total_matches = 0
    total_samples = 0
    times = []
    y_true = []
    y_pred = []

    results = []
    # Process each test sample
    for test_sample in test_json[:4]:
        start = time.time()
        test_audio_path = os.path.join(test_audio_dir, test_sample['audio_filepath'])
        
        # Load and process the test audio file
        waveform, sample_rate = load_audio(test_audio_path)
        embedding = get_wav2vec_embedding(waveform, sample_rate)
        retr_text = retrive_text(waveform, sample_rate)
        
        # Optional transcription, not needed for the FAISS search itself
        ids = torch.argmax(embedding, dim=-1)[0]
        transcription = processor.decode(ids)
        print(transcription)
        embedding_np = embedding.numpy()  # Shape: (1, hidden_size)

        # Use the configurable search function to find the nearest neighbor
        distances, indices = search_with_faiss(index, embedding_np, config)

        # Get nearest neighbor's index and metadata from the training dataset
        nearest_index = indices[0][0]
        nearest_distance = distances[0][0]
        nearest_train_metadata = train_metadata[nearest_index]

        # Compare the label of the test sample with the label of the nearest neighbor
        match = compare_label(test_sample['label'], nearest_train_metadata['label'])
        
        total_matches += match
        total_samples += 1

        print(f"Test sample {test_sample['id']} nearest neighbor: {nearest_train_metadata['id']}, "
              f"Label match: {'Yes' if match else 'No'}")
        y_true.append(test_sample['label'])
        y_pred.append(nearest_train_metadata['label'])
        # Track the time taken for this sample
        elapsed_time = time.time() - start
        times.append(elapsed_time)
        distance_threshold = 0.5
    
        if nearest_distance >= distance_threshold:
            if nearest_train_metadata['label'] in [4, 10]:
                result = {
                    "audio_filepath": test_sample['audio_filepath'],  # test audio file name
                    "text": retr_text,  # text from metadata
                    "label": nearest_train_metadata['label'],  # label from metadata
                    "attribute": extract_arg(retr_text)
                }
            else:
                result = {
                    "audio_filepath": test_sample['audio_filepath'],  # test audio file name
                    "text": nearest_train_metadata['text'],  # text from metadata
                    "label": nearest_train_metadata['label'],  # label from metadata
                    "attribute": nearest_train_metadata['attribute'],  # attribute from metadata
                }
            results.append(result)
        else:
            print(f"Test sample {test_sample['id']} skipped due to low distance: {nearest_distance:.4f}")


    # Calculate overall accuracy
    accuracy = total_matches / total_samples
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    print(f"Median processing time per sample: {np.median(times)} seconds")

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    with open(output_json_path, 'w') as output_file:
        json.dump(results, output_file, indent=4)

    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_directory', help="путь до папки с входными аудио")
    parser.add_argument('json_ann', help="путь до тестового json")
    parser.add_argument('output', help="путь выхода json")
    args = parser.parse_args()
    config = {
        'vector_dim': 768,  # Based on your embedding size
        'metric_type': 'l2',  # 'l2' for Euclidean distance
        'index_type': 'flat',  # Brute-force flat index
        'normalize': False,  # No normalization for L2
        'k': 1  # Retrieve only 1 nearest neighbor
    }
    # Parameters for indexing
    #audio_directory = "/home/user/rzhd_hack/split_for_model//train"
    #json_annotations = "/home/user/rzhd_hack/split_for_model/train.json"
    faiss_index_file = "./faiss_train_clear_ru_vec_small_2.index"
    labels_file = "./labels.pkl"
    metadata_file = "/home/user/rzhd_hack/split_for_model/train.pkl"
    #output_json_path = "/home/user/rzhd_hack/output.json"

    # Index audio and labels along with metadata for the training set
    #index_audio_files(audio_directory, json_annotations, faiss_index_file, labels_file, metadata_file)

    # Show first 4 vectors with metadata#
    #show_first_n_vectors_with_metadata(faiss_index_file, metadata_file, n=4)

    # Parameters for testing
    test_audio_dir = args.audio_directory
    test_json_data = args.json_ann    # Evaluate test set by finding nearest neighbor in the train vector store
    evaluate_test_set(test_audio_dir, test_json_data, faiss_index_file, labels_file, metadata_file,config,args.output)

