import TrainingModel
import transformers
import pickle
import tensorflow as tf
# Ignore the warning messages
import logging
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)

def convertTFTransformerToPT(saveNameBase):
    ptFormer = transformers.AutoModel.from_pretrained(saveNameBase, from_tf=True)
    ptFormer.save_pretrained(saveNameBase + "-PT")
    
    
    # with open('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/{}-Linear-Weights.pkl'.format(saveNameBase), 'rb') as fp:
    #     weights = pickle.load(fp)
    # TODO Add code for converting the linear weights into a torch linear layer


def splitAndStoreTFModelToDisk(transformerBase, weightsPath, visualDimensionSpace, saveNameBase):
    # Splits the Sentence Transformer and its linear layer
    # The Transformer can then be loaded into PT, and the linear weights can be added as a linear layer

    tokenizer = transformers.AutoTokenizer.from_pretrained(transformerBase)
    
    model = TrainingModel.SentenceModelWithLinearTransformation(transformerBase, visualDimensionSpace)
    
    import tensorflow as tf

    new_model = tf.keras.models.load_model('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/bert-base-arabertv2-Vit-B-32-1.keras')

    # Show the model architecture
    new_model.summary()
    # # print("="*100)
    # # print("len(model.get_weights())", len(model.get_weights()))
    # # print("="*100)
    # # model.set_weights(weightsPath)
    # model.load_weights(weightsPath).expect_partial()
    # # checkpoint = tf.train.Checkpoint(model)
    # # tf.train.Checkpoint.restore(checkpoint,save_path=weightsPath).expect_partial()
    saveNameBase = 'arabic-arabert-Vit-B-32'

    tokenizer.save_pretrained(saveNameBase + '-Tokenizer')
    model.transformer.save_pretrained(saveNameBase + '-Transformer')

    
    ptFormer = transformers.AutoModel.from_pretrained(saveNameBase + '-Transformer', from_tf=True)

    ptFormer.save_pretrained(saveNameBase + "-PT")
    
    # # model.push_to_hub(repo_id="pain/bert-base-arabertv2-Vit-B-32-using-tf_3")
    # linearWeights = model.postTransformation.get_weights()

    # import numpy as np
    # # Iterate through the layers and print the weights
    # for layer in model.layers:
    #     layer_name = layer.name
    #     layer_weights = layer.get_weights()
        
    #     if layer_weights:
    #         print(f"Layer: {layer_name}")
    #         for i, weights in enumerate(layer_weights):
    #             print(f"Weight {i}: {np.array(weights)}")
    #         print("\n")

    # print("="*100)
    # print("="*100)
    # print(linearWeights)
    # print("="*100)
    # print("="*100)
    # # print("Saving Linear Weights into pickle file.", linearWeights.shape)
    
    # with open('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/{}-Linear-Weights-laaaaaaaaaaaaatest.pkl'.format(saveNameBase), 'wb') as fp:
    #     pickle.dump(linearWeights, fp)


if __name__ == '__main__':
    weightsPath = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/old_files/aubmindlab_1/bert-base-arabertv2-Vit-B-32' # '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/multiple_checkpoints/bert-base-arabertv2-Vit-B-32-10'
    transformerBase = 'aubmindlab/bert-base-arabertv2'
    modelSaveBase = 'arabic-arabert-Vit-B-32'
    visualDimensionSpace = 512

    splitAndStoreTFModelToDisk(transformerBase, weightsPath, visualDimensionSpace, modelSaveBase)
    convertTFTransformerToPT(modelSaveBase + "-Transformer")
