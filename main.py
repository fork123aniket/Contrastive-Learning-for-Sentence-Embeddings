import tensorflow as tf
from tensorflow_addons.optimizers import AdamW, CyclicalLearningRate
from transformers import AutoConfig, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import numpy as np
import pandas as pd
from model import BertForCL


def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(1)
model_name_or_path, max_seq_length, adam_epsilon, embedding_dim = 'bert-base-uncased', 32, 1e-8, 768
weight_decay, num_train_epochs, batch_size, temp, pooler_type = 0.001, 10, 2, 0.05, 'cls_before_pooler'
num_sent, initial_learning_rate, maximal_learning_rate, factor, sent_to_comp = 2, 1e-4, 1e-2, 5, 0
assert pooler_type in ["cls", "cls_before_pooler"], "Unknown pooler type %s" % pooler_type
assert factor in np.arange(2, 9)

config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def process_batch(txt_list, tokenizer, max_len=max_seq_length):
    source_ls = [source for source, target in txt_list]
    target_ls = [target for source, target in txt_list]

    source_tokens = tokenizer(source_ls, truncation=True, padding="max_length", max_length=max_len)
    target_tokens = tokenizer(target_ls, truncation=True, padding="max_length", max_length=max_len)

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for i in range(len(source_tokens["input_ids"])):
        input_ids.append(source_tokens["input_ids"][i])
        input_ids.append(target_tokens["input_ids"][i])
        attention_mask.append(source_tokens["attention_mask"][i])
        attention_mask.append(target_tokens["attention_mask"][i])
        token_type_ids.append(source_tokens["token_type_ids"][i])
        token_type_ids.append(target_tokens["token_type_ids"][i])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask), tf.convert_to_tensor(token_type_ids)


dataset_df = pd.read_csv("/content/wiki1m_for_simcse.txt", names=["text"])
dataset_df.dropna(inplace=True)
source_texts = dataset_df["text"].values
target_texts = dataset_df["text"].values
data = list(zip(source_texts, target_texts))
input_ids, attention_mask, token_type_ids = process_batch(data, tokenizer, max_seq_length)

dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, token_type_ids))
dataset = (dataset.cache().repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE))
datasetGen = iter(dataset)

inp_id = tf.keras.layers.Input(shape=(input_ids.shape[1],), batch_size=batch_size, name='input_id', dtype='int32')
att_mask = tf.keras.layers.Input(shape=(attention_mask.shape[1],), batch_size=batch_size, name='masked_id', dtype='int32')
t_id = tf.keras.layers.Input(shape=(token_type_ids.shape[1],), batch_size=batch_size, name='input_token_id', dtype='int32')


def build_model(inputs):
    input_ids, attention_mask, token_type_id = inputs
    transformer_model = BertForCL('bert-base-uncased', config, pooler_type, num_sent, temp)
    output = transformer_model(input_ids, attention_mask, token_type_id)
    model = tf.keras.Model([input_ids, attention_mask, token_type_id], output)
    return model


model = build_model([inp_id, att_mask, t_id])
steps_per_epoch = (len(data) // batch_size) * num_train_epochs
clr = CyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                           maximal_learning_rate=maximal_learning_rate, scale_fn=lambda x: 1/(2.**(x-1)),
                           step_size=factor * steps_per_epoch)
optimizer = AdamW(weight_decay=weight_decay, learning_rate=clr, epsilon=adam_epsilon)
model.compile(optimizer, None)


def grad(model, inputs):
    with tf.GradientTape() as tape:
         output = model(inputs, training=True)
         loss_value = output[0]['loss']
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


for epoch in range(num_train_epochs):
    running_loss = 0.0
    for _ in range(input_ids.shape[0] // batch_size):
        (input_ids, attention_mask, token_type_ids) = next(datasetGen)
        loss_value, grads = grad(model, [input_ids, attention_mask, token_type_ids])
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_variables) if grad is not None)
        running_loss += loss_value
    print(f'epoch {epoch}, loss: {running_loss.numpy():.4f}')

sentences = [
    "chocolates are my favourite items.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "white chocolates and dark chocolates are favourites for many people.",
    "I love chocolates.",
    "Let me help you.",
    "There are some who influenced many.",
    "Chips are getting more popular these days.",
    "There are tools which help us get our work done.",
    "Electric vehicles are worth buying given their mileage on the road.",
    "NATO is the most powerful military alliance.",
    "Gone are the days when people got worry about their diets."
]

val_data = list(zip(sentences, sentences))
compare_sent = np.arange(len(sentences))
comp_indices = np.where(compare_sent != sent_to_comp)

input_ids, attention_mask, token_type_ids = process_batch(val_data, tokenizer, max_seq_length)
pool_output = tf.zeros([0, embedding_dim])
for batch in range(0, input_ids.shape[0], batch_size):
    outputs = model.predict([input_ids[batch:batch+batch_size], attention_mask[batch:batch+batch_size],
                             token_type_ids[batch:batch+batch_size]])
    pool_output = tf.concat([pool_output, outputs[1]["pooler_output"]], axis=0)
pool_output = tf.gather(pool_output, tf.range(0, pool_output.shape[0], num_sent))

mean_pooled = pool_output.numpy()
cosine_similarity([mean_pooled[sent_to_comp]], mean_pooled[comp_indices])
