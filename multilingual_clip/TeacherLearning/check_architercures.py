# import transformers


# modelBase = 'aubmindlab/bert-base-arabertv2'
# from_pt=True

# transformer = transformers.TFAutoModel.from_pretrained(modelBase, from_pt=from_pt)

# # Print the architecture summary
# print(transformer.summary())

# # Print the internal layer names
# for layer in transformer.layers[0].submodules:
#     print(layer.name)

import pickle

# # Open the pickle file in binary read mode
# pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/data/weights/Swedish-2M Linear Weights.pkl'  # Replace with the actual path to your pickle file
# with open(pickle_file_path, 'rb') as file:
#     loaded_content = pickle.load(file)
#     print(len(loaded_content))
#     print(loaded_content[0].shape)
#     print(loaded_content[1].shape)

# # Open the pickle file in binary read mode
# pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/data/weights/Swedish-2M Linear Weights.pkl'  # Replace with the actual path to your pickle file
# with open(pickle_file_path, 'rb') as file:
#     loaded_content = pickle.load(file)
#     print(len(loaded_content))
#     print(loaded_content[0].shape)
#     print(loaded_content[1].shape)

# # # Print the loaded content
# # print(loaded_content)


#check out checkpoints 

# Open the pickle file in binary read mode
# pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/multiple_checkpoints/arabic-arabert-Vit-B-32-Linear-Weights.pkl'
# pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/old_files/arabic-arabert-Vit-B-32-Linear-Weights.pkl'
# pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/old_files/aubmindlab_backup/Vit-B-32-Linear-Weights.pkl'
pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/multiple_checkpoints/postTransformation_layer_linear_latest.pickle'

with open(pickle_file_path, 'rb') as file:
    loaded_content = pickle.load(file)
    print(loaded_content)
    print(len(loaded_content))
    print(loaded_content[0].shape)
    print(loaded_content[1].shape)

# # Print the loaded content
# print(loaded_content)


# import tensorflow as tf

# new_model = tf.keras.models.load_model('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/bert-base-arabertv2-Vit-B-32-1.keras')

# # Show the model architecture
# new_model.summary()


# Error:
# --------
# (base) lenovo@lenovo-ThinkStation-P920:~/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning$  /usr/bin/env /bin/python3 /home/lenovo/.vscode/extensions/ms-python.python-2023.4.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 57737 -- /home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/check_architercures.py 
# 2023-08-07 14:41:30.963316: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
# 2023-08-07 14:41:31.008187: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
# 2023-08-07 14:41:31.008804: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2023-08-07 14:41:31.806246: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# 2023-08-07 14:41:33.214796: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1960] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
# Skipping registering GPU devices...
# Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertModel: ['cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'bert.embeddings.position_ids', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias', 'cls.predictions.transform.LayerNorm.weight']
# - This IS expected if you are initializing TFBertModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
# - This IS NOT expected if you are initializing TFBertModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
# All the weights of TFBertModel were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.
# ====================================================================================================
# ====================================================================================================
# postTransformation <keras.src.layers.core.dense.Dense object at 0x7f882e1c9eb0>
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================
# Training arg in SentenceModelWithLinearTransformation:  False
# Hellllllllllllllllllo
# 2
# Traceback (most recent call last):
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 521, in build
#     self.call(x, **kwargs)
#   File "/home/lenovo/.local/lib/python3.8/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
#     raise e.with_traceback(filtered_tb) from None
#   File "/tmp/__autograph_generated_filevv36zph4.py", line 13, in tf__call
#     ag__.ld(print)(ag__.converted_call(ag__.ld(type), (ag__.converted_call(ag__.ld(self).postTransformation, (ag__.converted_call(ag__.ld(self).generateMultipleEmbeddings, (ag__.ld(inputs), ag__.ld(training)), None, fscope),), None, fscope),), None, fscope))
#   File "/tmp/__autograph_generated_filelx6lrsns.py", line 12, in tf__generateMultipleEmbeddings
#     embs = ag__.converted_call(ag__.ld(self).transformer, ({'input_ids': ag__.ld(inds), 'attention_mask': ag__.ld(att)},), dict(training=ag__.ld(training)), fscope)['last_hidden_state']
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler
#     raise e.with_traceback(filtered_tb) from None
#   File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#     retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#   File "/tmp/__autograph_generated_filend9k0ss7.py", line 12, in tf__call
#     outputs = ag__.converted_call(ag__.ld(self).bert, (), dict(input_ids=ag__.ld(input_ids), attention_mask=ag__.ld(attention_mask), token_type_ids=ag__.ld(token_type_ids), position_ids=ag__.ld(position_ids), head_mask=ag__.ld(head_mask), inputs_embeds=ag__.ld(inputs_embeds), encoder_hidden_states=ag__.ld(encoder_hidden_states), encoder_attention_mask=ag__.ld(encoder_attention_mask), past_key_values=ag__.ld(past_key_values), use_cache=ag__.ld(use_cache), output_attentions=ag__.ld(output_attentions), output_hidden_states=ag__.ld(output_hidden_states), return_dict=ag__.ld(return_dict), training=ag__.ld(training)), fscope)
#   File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#     retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#   File "/tmp/__autograph_generated_filehq1f8qyt.py", line 127, in tf__call
#     embedding_output = ag__.converted_call(ag__.ld(self).embeddings, (), dict(input_ids=ag__.ld(input_ids), position_ids=ag__.ld(position_ids), token_type_ids=ag__.ld(token_type_ids), inputs_embeds=ag__.ld(inputs_embeds), past_key_values_length=ag__.ld(past_key_values_length), training=ag__.ld(training)), fscope)
#   File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#     ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#   File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#     inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)
# TypeError: in user code:

#     File "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/TrainingModel.py", line 72, in call  *
#         print(type(self.postTransformation(self.generateMultipleEmbeddings(inputs, training))))
#     File "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/TrainingModel.py", line 28, in generateMultipleEmbeddings  *
#         embs = self.transformer({'input_ids': inds, 'attention_mask': att}, training=training)['last_hidden_state']
#     File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler  **
#         raise e.with_traceback(filtered_tb) from None
#     File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#         retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#     File "/tmp/__autograph_generated_filend9k0ss7.py", line 12, in tf__call
#         outputs = ag__.converted_call(ag__.ld(self).bert, (), dict(input_ids=ag__.ld(input_ids), attention_mask=ag__.ld(attention_mask), token_type_ids=ag__.ld(token_type_ids), position_ids=ag__.ld(position_ids), head_mask=ag__.ld(head_mask), inputs_embeds=ag__.ld(inputs_embeds), encoder_hidden_states=ag__.ld(encoder_hidden_states), encoder_attention_mask=ag__.ld(encoder_attention_mask), past_key_values=ag__.ld(past_key_values), use_cache=ag__.ld(use_cache), output_attentions=ag__.ld(output_attentions), output_hidden_states=ag__.ld(output_hidden_states), return_dict=ag__.ld(return_dict), training=ag__.ld(training)), fscope)
#     File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#         retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#     File "/tmp/__autograph_generated_filehq1f8qyt.py", line 127, in tf__call
#         embedding_output = ag__.converted_call(ag__.ld(self).embeddings, (), dict(input_ids=ag__.ld(input_ids), position_ids=ag__.ld(position_ids), token_type_ids=ag__.ld(token_type_ids), inputs_embeds=ag__.ld(inputs_embeds), past_key_values_length=ag__.ld(past_key_values_length), training=ag__.ld(training)), fscope)
#     File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#         ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#     File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#         inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)

#     TypeError: Exception encountered when calling layer 'tf_bert_model' (type TFBertModel).
    
#     in user code:
    
#         File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/modeling_tf_utils.py", line 1061, in run_call_with_unpacked_inputs  *
#             return func(self, **unpacked_inputs)
#         File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/models/bert/modeling_tf_bert.py", line 1088, in call  *
#             outputs = self.bert(
#         File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler  **
#             raise e.with_traceback(filtered_tb) from None
#         File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#             retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#         File "/tmp/__autograph_generated_filehq1f8qyt.py", line 127, in tf__call
#             embedding_output = ag__.converted_call(ag__.ld(self).embeddings, (), dict(input_ids=ag__.ld(input_ids), position_ids=ag__.ld(position_ids), token_type_ids=ag__.ld(token_type_ids), inputs_embeds=ag__.ld(inputs_embeds), past_key_values_length=ag__.ld(past_key_values_length), training=ag__.ld(training)), fscope)
#         File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#             ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#         File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#             inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)
    
#         TypeError: Exception encountered when calling layer 'bert' (type TFBertMainLayer).
        
#         in user code:
        
#             File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/modeling_tf_utils.py", line 1061, in run_call_with_unpacked_inputs  *
#                 return func(self, **unpacked_inputs)
#             File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/models/bert/modeling_tf_bert.py", line 780, in call  *
#                 embedding_output = self.embeddings(
#             File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler  **
#                 raise e.with_traceback(filtered_tb) from None
#             File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#                 ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#             File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#                 inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)
        
#             TypeError: Exception encountered when calling layer 'embeddings' (type TFBertEmbeddings).
            
#             in user code:
            
#                 File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/models/bert/modeling_tf_bert.py", line 203, in call  *
#                     inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
            
#                 TypeError: Value passed to parameter 'indices' has DataType float32 not in list of allowed values: int32, int64
            
            
#             Call arguments received by layer 'embeddings' (type TFBertEmbeddings):
#               • input_ids=tf.Tensor(shape=(None, 32), dtype=float32)
#               • position_ids=None
#               • token_type_ids=tf.Tensor(shape=(None, 32), dtype=int32)
#               • inputs_embeds=None
#               • past_key_values_length=0
#               • training=False
        
        
#         Call arguments received by layer 'bert' (type TFBertMainLayer):
#           • input_ids=tf.Tensor(shape=(None, 32), dtype=float32)
#           • attention_mask=tf.Tensor(shape=(None, 32), dtype=float32)
#           • token_type_ids=None
#           • position_ids=None
#           • head_mask=None
#           • inputs_embeds=None
#           • encoder_hidden_states=None
#           • encoder_attention_mask=None
#           • past_key_values=None
#           • use_cache=True
#           • output_attentions=False
#           • output_hidden_states=False
#           • return_dict=True
#           • training=False
    
    
#     Call arguments received by layer 'tf_bert_model' (type TFBertModel):
#       • input_ids={'input_ids': 'tf.Tensor(shape=(None, 32), dtype=float32)', 'attention_mask': 'tf.Tensor(shape=(None, 32), dtype=float32)'}
#       • attention_mask=None
#       • token_type_ids=None
#       • position_ids=None
#       • head_mask=None
#       • inputs_embeds=None
#       • encoder_hidden_states=None
#       • encoder_attention_mask=None
#       • past_key_values=None
#       • use_cache=None
#       • output_attentions=None
#       • output_hidden_states=None
#       • return_dict=None
#       • training=False


# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/check_architercures.py", line 58, in <module>
#     new_model = tf.keras.models.load_model('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/bert-base-arabertv2-Vit-B-32-1.keras')
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/saving/saving_api.py", line 230, in load_model
#     return saving_lib.load_model(
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/saving/saving_lib.py", line 275, in load_model
#     raise e
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/saving/saving_lib.py", line 240, in load_model
#     model = deserialize_keras_object(
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/saving/serialization_lib.py", line 707, in deserialize_keras_object
#     instance.build_from_config(build_config)
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/engine/base_layer.py", line 2341, in build_from_config
#     self.build(input_shape)
#   File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 523, in build
#     raise ValueError(
# ValueError: You cannot build your model by calling `build` if your layers do not support float type inputs. Instead, in order to instantiate and build your model, call your model on real tensor data (of the correct dtype).

# The actual error from `call` is: in user code:

#     File "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/TrainingModel.py", line 72, in call  *
#         print(type(self.postTransformation(self.generateMultipleEmbeddings(inputs, training))))
#     File "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/TrainingModel.py", line 28, in generateMultipleEmbeddings  *
#         embs = self.transformer({'input_ids': inds, 'attention_mask': att}, training=training)['last_hidden_state']
#     File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler  **
#         raise e.with_traceback(filtered_tb) from None
#     File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#         retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#     File "/tmp/__autograph_generated_filend9k0ss7.py", line 12, in tf__call
#         outputs = ag__.converted_call(ag__.ld(self).bert, (), dict(input_ids=ag__.ld(input_ids), attention_mask=ag__.ld(attention_mask), token_type_ids=ag__.ld(token_type_ids), position_ids=ag__.ld(position_ids), head_mask=ag__.ld(head_mask), inputs_embeds=ag__.ld(inputs_embeds), encoder_hidden_states=ag__.ld(encoder_hidden_states), encoder_attention_mask=ag__.ld(encoder_attention_mask), past_key_values=ag__.ld(past_key_values), use_cache=ag__.ld(use_cache), output_attentions=ag__.ld(output_attentions), output_hidden_states=ag__.ld(output_hidden_states), return_dict=ag__.ld(return_dict), training=ag__.ld(training)), fscope)
#     File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#         retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#     File "/tmp/__autograph_generated_filehq1f8qyt.py", line 127, in tf__call
#         embedding_output = ag__.converted_call(ag__.ld(self).embeddings, (), dict(input_ids=ag__.ld(input_ids), position_ids=ag__.ld(position_ids), token_type_ids=ag__.ld(token_type_ids), inputs_embeds=ag__.ld(inputs_embeds), past_key_values_length=ag__.ld(past_key_values_length), training=ag__.ld(training)), fscope)
#     File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#         ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#     File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#         inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)

#     TypeError: Exception encountered when calling layer 'tf_bert_model' (type TFBertModel).
    
#     in user code:
    
#         File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/modeling_tf_utils.py", line 1061, in run_call_with_unpacked_inputs  *
#             return func(self, **unpacked_inputs)
#         File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/models/bert/modeling_tf_bert.py", line 1088, in call  *
#             outputs = self.bert(
#         File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler  **
#             raise e.with_traceback(filtered_tb) from None
#         File "/tmp/__autograph_generated_fileud9ch37t.py", line 37, in tf__run_call_with_unpacked_inputs
#             retval_ = ag__.converted_call(ag__.ld(func), (ag__.ld(self),), dict(**ag__.ld(unpacked_inputs)), fscope)
#         File "/tmp/__autograph_generated_filehq1f8qyt.py", line 127, in tf__call
#             embedding_output = ag__.converted_call(ag__.ld(self).embeddings, (), dict(input_ids=ag__.ld(input_ids), position_ids=ag__.ld(position_ids), token_type_ids=ag__.ld(token_type_ids), inputs_embeds=ag__.ld(inputs_embeds), past_key_values_length=ag__.ld(past_key_values_length), training=ag__.ld(training)), fscope)
#         File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#             ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#         File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#             inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)
    
#         TypeError: Exception encountered when calling layer 'bert' (type TFBertMainLayer).
        
#         in user code:
        
#             File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/modeling_tf_utils.py", line 1061, in run_call_with_unpacked_inputs  *
#                 return func(self, **unpacked_inputs)
#             File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/models/bert/modeling_tf_bert.py", line 780, in call  *
#                 embedding_output = self.embeddings(
#             File "/home/lenovo/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler  **
#                 raise e.with_traceback(filtered_tb) from None
#             File "/tmp/__autograph_generated_filesb167eyx.py", line 41, in tf__call
#                 ag__.if_stmt((ag__.ld(input_ids) is not None), if_body_1, else_body_1, get_state_1, set_state_1, ('inputs_embeds',), 1)
#             File "/tmp/__autograph_generated_filesb167eyx.py", line 36, in if_body_1
#                 inputs_embeds = ag__.converted_call(ag__.ld(tf).gather, (), dict(params=ag__.ld(self).weight, indices=ag__.ld(input_ids)), fscope)
        
#             TypeError: Exception encountered when calling layer 'embeddings' (type TFBertEmbeddings).
            
#             in user code:
            
#                 File "/home/lenovo/.local/lib/python3.8/site-packages/transformers/models/bert/modeling_tf_bert.py", line 203, in call  *
#                     inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
            
#                 TypeError: Value passed to parameter 'indices' has DataType float32 not in list of allowed values: int32, int64
            
            
#             Call arguments received by layer 'embeddings' (type TFBertEmbeddings):
#               • input_ids=tf.Tensor(shape=(None, 32), dtype=float32)
#               • position_ids=None
#               • token_type_ids=tf.Tensor(shape=(None, 32), dtype=int32)
#               • inputs_embeds=None
#               • past_key_values_length=0
#               • training=False
        
        
#         Call arguments received by layer 'bert' (type TFBertMainLayer):
#           • input_ids=tf.Tensor(shape=(None, 32), dtype=float32)
#           • attention_mask=tf.Tensor(shape=(None, 32), dtype=float32)
#           • token_type_ids=None
#           • position_ids=None
#           • head_mask=None
#           • inputs_embeds=None
#           • encoder_hidden_states=None
#           • encoder_attention_mask=None
#           • past_key_values=None
#           • use_cache=True
#           • output_attentions=False
#           • output_hidden_states=False
#           • return_dict=True
#           • training=False
    
    
#     Call arguments received by layer 'tf_bert_model' (type TFBertModel):
#       • input_ids={'input_ids': 'tf.Tensor(shape=(None, 32), dtype=float32)', 'attention_mask': 'tf.Tensor(shape=(None, 32), dtype=float32)'}
#       • attention_mask=None
#       • token_type_ids=None
#       • position_ids=None
#       • head_mask=None
#       • inputs_embeds=None
#       • encoder_hidden_states=None
#       • encoder_attention_mask=None
#       • past_key_values=None
#       • use_cache=None
#       • output_attentions=None
#       • output_hidden_states=None
#       • return_dict=None
#       • training=False