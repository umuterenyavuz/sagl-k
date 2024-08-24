import json
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForTokenClassification
from sklearn.model_selection import train_test_split

def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            labels.append(data['label'])
    return texts, labels

def tokenize_and_align_labels(texts, labels, tokenizer, max_len=128):
    tokenized_inputs = tokenizer(texts, max_length=max_len, truncation=True, padding='max_length', return_offsets_mapping=True, is_split_into_words=False)
    
    aligned_labels = []
    for i, label_list in enumerate(labels):
        offsets = tokenized_inputs.offset_mapping[i]
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])
        
        for label_start, label_end, label_type in label_list:
            for idx, (start, end) in enumerate(offsets):
                if start >= label_start and end <= label_end:
                    label_ids[idx] = label_type
        
        aligned_labels.append(label_ids)
    
    # Debug: Check shapes and types of tokenized_inputs and aligned_labels
    print("Tokenized Inputs:", tokenized_inputs)
    print("Aligned Labels:", aligned_labels)
    
    return tokenized_inputs, aligned_labels

def create_tf_dataset(tokenized_inputs, aligned_labels, batch_size=8):
    input_ids = tf.convert_to_tensor(tokenized_inputs['input_ids'])
    attention_mask = tf.convert_to_tensor(tokenized_inputs['attention_mask'])
    token_type_ids = tf.convert_to_tensor(tokenized_inputs['token_type_ids'])
    labels = tf.convert_to_tensor(aligned_labels, dtype=tf.int32)
    
    # Debug: Check tensor shapes
    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)
    print("Token Type IDs Shape:", token_type_ids.shape)
    print("Labels Shape:", labels.shape)
    
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids},
        {'labels': labels}
    ))
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset

def main():
    # Load data
    texts, labels = load_data('all.jsonl')

    # Initialize fast tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    num_labels = len(set(lbl[2] for sublist in labels for lbl in sublist if lbl[2] != -100)) + 1
    model = TFBertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # Tokenize and align labels
    tokenized_inputs, aligned_labels = tokenize_and_align_labels(texts, labels, tokenizer)

    # Create TensorFlow dataset
    train_dataset = create_tf_dataset(tokenized_inputs, aligned_labels)

    # Compile and train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=model.compute_loss)
    model.fit(train_dataset, epochs=3)

if __name__ == "__main__":
    main()