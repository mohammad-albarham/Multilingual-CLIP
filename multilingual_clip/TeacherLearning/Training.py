import wandb
from wandb.keras import WandbMetricsLogger

import Dataset, TrainingModel
import tensorflow as tf
import transformers
import datasets
import Utils
import datetime
import pickle

from logger import logger
# from tensorflow.keras import mixed_precision

# mixed_precision.set_global_policy('mixed_float16')


def loadTargetEmbeddings(dataset_name, validationSize=5000): # 2000000

    trainSamples = datasets.load_dataset(dataset_name, split='train[{}:]'.format(validationSize))
    valSamples = datasets.load_dataset(dataset_name, split='train[:{}]'.format(validationSize))

    
    logger.info(f"len(trainSamples): {len(trainSamples)}") # len(trainSamples) 1995000
    logger.info(f"len(valSamples): {len(valSamples)}") # len(valSamples) 5000

    embeddingShape = tf.convert_to_tensor(trainSamples[0]['embedding']).shape # (1, 640)

    logger.info(f"embeddingShape of one of the embeddings of the trainsamples: {embeddingShape}")


    return trainSamples, valSamples, embeddingShape



def singleGPUTraining():

    # Tune the hyperparameter 
    stepsPerEpoch, lr = 7813, 0.00001  #1133 # 10, 0.00005 # 1172, 0.00005  # 8851, 0.00005 # 2213 # 566405/128 = 4425.0390625 # 586, 0.00001 # maximum number of stepPerEpoch I can feed: 585.9375
    gradAccumSteps, batchSize = 1, 128 # 1, 2 # 1, 128 # 256
    epochs = 1000
    numTrainSteps, numWarmupSteps = 1562600, 1000 # 1
    
    modelBase = 'aubmindlab/bert-large-arabertv2' # 'xlm-roberta-large' # 'bert-base-multilingual-cased'  # 'aubmindlab/bert-base-arabertv2'
    tokenizerBase = 'aubmindlab/bert-large-arabertv2' # 'xlm-roberta-large' #'bert-base-multilingual-cased' # 'aubmindlab/bert-base-arabertv2'
    imageBase = "Vit-B-16-plus-240"
    modelName = "bert-large-arabertv2" +  "-" + imageBase + "-" # modelBase  + "-" + imageBase + "-" # '{}-{}'.format(modelBase, imageBase) # # modelName = modelBase.split("/")[1]  + "-" + imageBase + "-{}" # '{}-{}'.format(modelBase, imageBase)

    log_name =  "bert-large-arabertv2" +  "-" + imageBase + "-" # modelBase  + "-" + imageBase + "-"
    
    startWeights = "/home/lenovo/Desktop/arabic_clip/arabert_v2_vit_B_16_plus/phase_1/bert-large-arabertv2-Vit-B-16-plus-240- 36_2023 10 08 - 02 45 18_epoch_36_internal_.keras" # None # "/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/old_files/aubmindlab_1/bert-base-arabertv2-Vit-B-32"

    dataset_name = "Arabic-Clip/Arabic_dataset_1M_translated_jsonl_format_ViT-B-16-plus-240" # "Arabic-Clip/Arabic_dataset_3M_translated_cleaned_v2_jsonl_format_ViT-B-16-plus-240"
    
    trainEmbeddings, valEmbeddings, imageEncoderDimensions = loadTargetEmbeddings(dataset_name=dataset_name)

    def createOptimizerFunc():
        optimizer, schedule = transformers.optimization_tf.create_optimizer(lr, numTrainSteps, numWarmupSteps)
        if (gradAccumSteps <= 1):
            return optimizer
        else:
            return Utils.GradientAccumulator(optimizer, gradAccumSteps)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerBase)

    logger.info(f"imageEncoderDimensions[-1]: {imageEncoderDimensions[-1]}")

    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, imageEncoderDimensions[-1])

    # from tensorflow import keras
    # model = keras.models.load_model("/home/lenovo/Desktop/arabic_clip/arabert_v2_vit_B_16_plus/phase_1/bert-large-arabertv2-Vit-B-16-plus-240- 36_2023 10 08 - 02 45 18_epoch_36_internal_")
    # model.summary()  # Examine the model's architecture

    logger.info("Finishing loading keras checkpoint")

    


    if (startWeights is not None):

        logger.info("Loading weights ...")
        model.load_weights(startWeights,skip_mismatch=True)

        # pickle_file_path = "/home/lenovo/Desktop/arabic_clip/arabert_v2_vit_B_16_plus/phase_1/heads_of_the_model_bert-large-arabertv2-Vit-B-16-plus-240-36_.pickle"

        # logger.info("Finishing load the weights except dense layer")
        # logger.info("Start loading the dense layer weights")
        # with open(pickle_file_path, 'rb') as pickle_file:
        #     loaded_weights = pickle.load(pickle_file)
        #     logger.info(len(loaded_weights))
        #     logger.info(type(loaded_weights))
        #     logger.info(type(loaded_weights[0]))
        #     logger.info(type(loaded_weights[1]))
        #     logger.info(len(loaded_weights[0]))
        #     logger.info(len(loaded_weights[1]))
        #     logger.info("embeddingSize")

        # # logger.info("Finish loading the dense layer weights")
        # # model = tf.keras.models.load_model(startWeights)
        # # https://github.com/keras-team/keras/issues/7229
        # # https://keras.io/api/saving/weights_saving_and_loading/

        # # Define the desired shape
        # batch_size = None  # This represents the variable batch size
        # feature_size = 1024  # This represents the feature size

        # a_out = model.postTransformation(tf.convert_to_tensor(tf.ones((128, feature_size), dtype=tf.float32)))

        # model.postTransformation.set_weights([loaded_weights[0], loaded_weights[1]])


    model.compile(createOptimizerFunc(), loss='mse', metrics=['mae', 'cosine_similarity']) # I added the loss argument

    trainDataset, valDataset = Dataset.createTrainingAndValidationDataset(trainEmbeddings, 
                                                                          valEmbeddings, 
                                                                          batchSize,
                                                                          tokenizer,
                                                                        #   targetCaptions=targetCaptions,
                                                                          maxSeqLen = 64,
                                                                          encoderDims=imageEncoderDimensions)


    if (gradAccumSteps > 1):  # In order to make fair logging on Wandb
        stepsPerEpoch *= gradAccumSteps


    # Print the architecture summary

    logger.info(model.postTransformation.get_weights())


    # Print layer names and their weights
    for layer in model.layers:
        logger.info(layer)
        if hasattr(layer, 'weights'):
            weights = layer.get_weights()
            for i, weight_array in enumerate(weights):
                logger.info(f"Layer: {layer.name}, Weight Array {i + 1}: {weight_array.shape}")



    #TODO Adding the logs to TensorBoard

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y %m %d - %H %M %S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="epoch")


    #### Configure the WandB
    display_name = "experiment-" + log_name +  datetime.datetime.now().strftime("%Y %m %d - %H %M %S")

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
            "imageBase": imageBase,
            "Dataset": dataset_name
        },
    )

    logger.info("Start model.fit")
    logger.info(f"trainDataset sample: {next(iter(trainDataset))}")

    # checkpoint_filepath = "model-{epoch:02d}-{val_loss:.2f}"
    filepath = 'model_wandb'

    model.fit(trainDataset, epochs=epochs, steps_per_epoch=stepsPerEpoch,
              validation_data=valDataset,
              callbacks=[
                  Utils.CustomSaveCallBack(modelName, saveInterval=1,firstSavePoint=0, log_name=log_name,tokenizer=tokenizer,model=model),
                #   WandbModelCheckpoint(filepath = filepath, verbose=1, save_freq='epoch', save_best_only=True), #save_freq='epoch'
                  WandbMetricsLogger(log_freq="batch"), # epoch
                  # tensorboard_callback,
                  
                # f"{model_config['MODEL_DIR']}/{exp_id}-model-fold{fold_num}-best.h5", "keras_cifar10_{epoch:02d}"
              ],
            #   verbose=0,
            #   workers=1
            )
    
    logger.info("End model.fit")


    logger.info("Saving model ......................")

    
    logger.info("Calling model.summary")

    logger.info(model.postTransformation.get_weights())
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logger.info(short_model_summary)


    wandb.finish()

if __name__ == '__main__':
    # https://www.tensorflow.org/guide/keras/distributed_training#introduction
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
    	singleGPUTraining()
