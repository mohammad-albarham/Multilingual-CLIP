import TrainingModel
import transformers
import pickle
import tensorflow as tf
# Ignore the warning messages
import logging
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)

def convertTFTransformerToPT(saveNameBase):
    ptFormer = transformers.AutoModel.from_pretrained(saveNameBase, from_tf=True)
    ptFormer.save_pretrained(saveNameBase + '-Transformer' + "-PT")
    
    
    # with open('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/{}-Linear-Weights.pkl'.format(saveNameBase), 'rb') as fp:
    #     weights = pickle.load(fp)
    # TODO Add code for converting the linear weights into a torch linear layer


def splitAndStoreTFModelToDisk(transformerBase, weightsPath, visualDimensionSpace, saveNameBase):
    # Splits the Sentence Transformer and its linear layer
    # The Transformer can then be loaded into PT, and the linear weights can be added as a linear layer

    tokenizer = transformers.AutoTokenizer.from_pretrained(transformerBase)
    
    model = TrainingModel.SentenceModelWithLinearTransformation(transformerBase, visualDimensionSpace)
    
    # print("="*100)
    # print("len(model.get_weights())", len(model.get_weights()))
    # print("="*100)
    # model.set_weights(weightsPath)
    # model.load_weights(weightsPath).expect_partial()
    checkpoint = tf.train.Checkpoint(model)
    tf.train.Checkpoint.restore(checkpoint,save_path=weightsPath).expect_partial()

    tokenizer.save_pretrained(saveNameBase + '-Tokenizer')
    model.transformer.save_pretrained(saveNameBase + '-Transformer')
    linearWeights = model.postTransformation.get_weights()
    # print("Saving Linear Weights into pickle file.", linearWeights.shape)
    
    with open('/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/{}-Linear-Weights.pkl'.format(saveNameBase), 'wb') as fp:
        pickle.dump(linearWeights, fp)


if __name__ == '__main__':
    weightsPath = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/aubmindlab/bert-base-arabertv2-Vit-B-32.index'
    transformerBase = 'aubmindlab/bert-base-arabertv2'
    modelSaveBase = 'Vit-B-32'
    visualDimensionSpace = 512

    splitAndStoreTFModelToDisk(transformerBase, weightsPath, visualDimensionSpace, modelSaveBase)
    convertTFTransformerToPT(modelSaveBase + "-Transformer")
