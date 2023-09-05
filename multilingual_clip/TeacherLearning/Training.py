import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import random

import Dataset, TrainingModel
import tensorflow as tf
import transformers
import datasets
import Utils
import datetime

# def loadTextTranslations():
#     dataset_Translations_arabic = datasets.load_dataset('Arabic-Clip/mscoco_2014_en_ar_mapping')['train']
#     print("="*100)
#     print("len(dataset_Translations_arabic)", len(dataset_Translations_arabic))
#     print("="*100)
#     return dataset_Translations_arabic

def loadTargetEmbeddings(imageBase="Vit-B-32"):

    print("Start loading the embeddings ..... ")
    # validationSize = 1
    
    trainSamples = datasets.load_dataset('Arabic-Clip/mscoco_jsonl_full', imageBase,
                                         split='train')
    valSamples = datasets.load_dataset('Arabic-Clip/mscoco_jsonl_full', imageBase,
                                       split='validation') # 

    print("="*100)
    print("len(trainSamples)", len(trainSamples)) # len(trainSamples) 1995000
    print("len(valSamples)", len(valSamples)) # len(valSamples) 5000

    embeddingShape = tf.convert_to_tensor(trainSamples[0]['embedding']).shape # (1, 512)

    print("embeddingShape of one of the embeddings of the trainsamples: ", embeddingShape)
    print("="*100)

    print("End loading the embeddings ..... ")

    return trainSamples, valSamples, embeddingShape


def singleGPUTraining():
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    # numValidationSamples = 5000 
    stepsPerEpoch, lr = 2213, 0.00001 # 2213 # 566405/256 = 2212.51953125 # 586, 0.00001 # maximum number of stepPerEpoch I can feed: 585.9375
    gradAccumSteps, batchSize = 1, 256 # 256
    numTrainSteps, numWarmupSteps = 99999999, 1000 # 1

    modelBase = 'aubmindlab/bert-base-arabertv2'
    tokenizerBase = 'aubmindlab/bert-base-arabertv2'
    imageBase = "Vit-B-32"
    modelName = "bert-base-arabertv2-Vit-B-32-{}" # '{}-{}'.format(modelBase, imageBase)

    startWeights = None # "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/old_files/aubmindlab_1/bert-base-arabertv2-Vit-B-32"

    # targetCaptions = loadTextTranslations()

    # print("")
    
    trainEmbeddings, valEmbeddings, imageEncoderDimensions = loadTargetEmbeddings(imageBase=imageBase)

    def createOptimizerFunc():
        optimizer, schedule = transformers.optimization_tf.create_optimizer(lr, numTrainSteps, numWarmupSteps)
        if (gradAccumSteps <= 1):
            return optimizer
        else:
            return Utils.GradientAccumulator(optimizer, gradAccumSteps)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerBase)

    print("="*100)
    print("imageEncoderDimensions[-1]: ", imageEncoderDimensions[-1])
    print("="*100)

    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, imageEncoderDimensions[-1])

    if (startWeights is not None):
        print("="*100)
        print("Loading weights ...")
        model.load_weights(startWeights)
        print("="*100)



    model.compile(createOptimizerFunc(), loss='mse', metrics=['mae', 'cosine_similarity']) # I added the loss argument

    trainDataset, valDataset = Dataset.createTrainingAndValidationDataset(trainEmbeddings, 
                                                                          valEmbeddings, 
                                                                          batchSize,
                                                                          tokenizer,
                                                                        #   targetCaptions=None,
                                                                          maxSeqLen = 64,
                                                                          encoderDims=imageEncoderDimensions)


    # from datasets import push_to_hub

    # push_to_hub(trainDataset,"pain/training-dataset-arabic-teacher-learning")

    if (gradAccumSteps > 1):  # In order to make fair logging on Wandb
        stepsPerEpoch *= gradAccumSteps

    # print("="*500)
    # print(model.postTransformation.get_weights())
    # print("="*500)
    # # Specify the path where you want to save the pickle file
    # pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/multiple_checkpoints/postTransformation_layer.pickle'

    # # Save the layer using pickle
    # with open(pickle_file_path, 'wb') as pickle_file:
    #     pickle.dump(layer_to_save, pickle_file)

    # Print the architecture summary
    print("="*500)
    # print(model.summary())

    # print(model.postTransformation.get_layer)
    print(model.postTransformation.get_weights())

    # Print the internal layer names
    # print("Print the internal layer names")
    # for layer in model.layers[1].submodules:
    #     print(layer.name)

    # # Print the layer names
    # for layer_name in model.layers:
    #     print(layer_name.name)

    # Print layer names and their weights
    for layer in model.layers:
        print(layer)
        if hasattr(layer, 'weights'):
            weights = layer.get_weights()
            for i, weight_array in enumerate(weights):
                print(f"Layer: {layer.name}, Weight Array {i + 1}: {weight_array.shape}")

    # # Access the dense layer and get its weights
    # dense_layer = model.layers[-1]  # Assuming the dense layer is the last layer
    # dense_weights = dense_layer.get_weights()

    #     # Access the weights of the postTransformation dense layer
    # dense_layer = model.layers[-1].postTransformation
    # dense_weights = dense_layer.get_weights()
    # Access the weights of the postTransformation dense layer using TensorFlow graph
    # graph = tf.compat.v1.get_default_graph()
    # dense_weights = graph.get_tensor_by_name('tf_bert_model/postTransformation/kernel:0')

    # # Print the weights of the dense layer
    # print(dense_weights)

    print("="*500)

    #TODO Adding the logs to TensorBoard

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y %m %d - %H %M %S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="epoch")



    #### Configure the WandB
    display_name = "experiment-2023-09-04-arabert-base-Vit-32-" +  datetime.datetime.now().strftime("%Y %m %d - %H %M %S")

        # Start a run, tracking hyperparameters
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="mscoco_teacher_learning_full_data",

        name=display_name,

        # track hyperparameters and run metadata with wandb.config
        config={
            "stepsPerEpoch": stepsPerEpoch,
            "lr": lr,
            "gradAccumSteps": gradAccumSteps,
            "batchSize": batchSize,
            "numTrainSteps": numTrainSteps,
            "numWarmupSteps": numWarmupSteps,
            "loss": "mse",
            "metrics": "mae, cosine_similarity",
            "modelBase": modelBase,
            "tokenizerBase": tokenizerBase,
            "imageBase": imageBase
        },
    )

    print("Start model.fit")
    print("trainDataset sample: ", next(iter(trainDataset)))

    model.fit(trainDataset, epochs=1, steps_per_epoch=stepsPerEpoch,
              validation_data=valDataset,
              callbacks=[
                  Utils.CustomSaveCallBack(modelName, saveInterval=1, firstSavePoint=1),
                  tensorboard_callback,
                  WandbMetricsLogger(log_freq="batch"),
                  WandbModelCheckpoint(filepath="bert-base-arabertv2-Vit-B-32-epoch_{epoch:02d}-val_loss_{val_loss:.2f}-wandb-model-" + datetime.datetime.now().strftime("%Y %m %d - %H %M %S"), monitor="val_loss", verbose=1),
              ],
            #   verbose=0,
              workers=112
              )
    
    print("End model.fit")


    print("="*100)
    print("Saving model ......................")
    saveNameBase = 'arabic-arabert-Vit-B-32' + datetime.datetime.now().strftime("%Y %m %d - %H %M %S")

    tokenizer.save_pretrained(saveNameBase + '-Tokenizer-after-finish-training')
    model.transformer.save_pretrained(saveNameBase + '-Transformer-after-finish-training')

    
    # ptFormer = transformers.AutoModel.from_pretrained(saveNameBase + '-Transformer-after-finish-training', from_tf=True, device_map="cpu")

    # ptFormer.save_pretrained(saveNameBase + "-PT")

    print("Saving model ......................")
    
    print("="*100)
    print("Calling model.summary")
    print(model.postTransformation.get_weights())
    import pickle
    # Save the layer using pickle
    pickle_file_path = '/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/checkpoints_mscoco_pickle/postTransformation_layer_linear_latest_after_finish_training_' + datetime.datetime.now().strftime("%Y %m %d - %H %M %S") + "_.pickle"
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(model.postTransformation.get_weights(), pickle_file)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)
    print("="*100)

    # wandb.alert(
    # title="The training is finished"
    # )

    wandb.finish()

if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
    	singleGPUTraining()
