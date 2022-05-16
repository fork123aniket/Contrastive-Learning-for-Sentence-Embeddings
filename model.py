import tensorflow as tf
from transformers.models.bert.modeling_tf_bert import TFBertPreTrainedModel, TFBertModel
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput, TFBaseModelOutputWithPoolingAndCrossAttentions


def cl_forward(cls, encoder, input_ids=None, attention_mask=None, token_type_ids=None,
               pooler_type='cls', num_sent=2, temp=0.05):

    return_dict = cls.config.use_return_dict
    batch_size = input_ids.shape[0] // num_sent

    outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      return_dict=True, output_hidden_states=False if pooler_type == 'cls' else True)

    pooler_output = outputs.last_hidden_state[:, 0]
    pooler_output = tf.reshape(pooler_output, (batch_size, num_sent, pooler_output.shape[-1]))

    if pooler_type == "cls":
        pooler_output = tf.keras.layers.Dense(cls.config.hidden_size, 'tanh')(pooler_output)

    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    cos_sim = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
    cos_sim = cos_sim(z1, z2) / temp
    labels = tf.cast(tf.range(cos_sim.shape[0]), tf.int64)
    loss = tf.keras.losses.CategoricalCrossentropy()(labels, cos_sim)

    emb_outputs = encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      return_dict=True, output_hidden_states=False)

    emb_pooler_output = emb_outputs.last_hidden_state[:, 0]
    if pooler_type == "cls":
        emb_pooler_output = tf.keras.layers.Dense(cls.config.hidden_size, 'tanh')(emb_pooler_output)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return (((loss,) + output) if loss is not None else output), (emb_outputs[0], emb_pooler_output) + emb_outputs[2:]

    return (TFSequenceClassifierOutput(loss=loss, logits=cos_sim, hidden_states=outputs.hidden_states,
                                       attentions=outputs.attentions),
            TFBaseModelOutputWithPoolingAndCrossAttentions(pooler_output=emb_pooler_output,
                                                           last_hidden_state=emb_outputs.last_hidden_state,
                                                           hidden_states=emb_outputs.hidden_states))


class BertForCL(TFBertPreTrainedModel):

    def __init__(self, model_name_or_path, config, pooler_type, num_sent, temp):
        super().__init__(config)
        self.bert = TFBertModel.from_pretrained(model_name_or_path, config=config)
        self.pooler_type = pooler_type
        self.num_sent = num_sent
        self.temp = temp

    def __call__(self, input_ids=None, attention_mask=None, token_type_ids=None):
        return cl_forward(self, self.bert, input_ids, attention_mask,
                          token_type_ids, self.pooler_type, self.num_sent,
                          self.temp)
