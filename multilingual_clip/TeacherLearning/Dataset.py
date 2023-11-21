import tensorflow as tf
from logger import logger

def createDataset(embeddings, batchSize, tokenizer, maxSeqLen=32, loopForever=True,
                  shuffleSize=None, encoderDims=(1, 768),precision_16=False):
    
    precision_settings = tf.float16 if precision_16 else tf.float32

    def generatorFunc():
        while True:
            embeddings.shuffle()

            for d in embeddings:
                textEmb,caption = d['embeddings'], d["ar_caption"]

                if (caption is None):
                    logger.info("Skipping example due to None caption")
                    continue

                textIds = tokenizer.encode(caption)

                seqLen = len(textIds)

                if (seqLen > maxSeqLen):
                    logger.info("Skipping example due to max length")
                    continue

                padSize = maxSeqLen - len(textIds)
                textIds = textIds + [0] * padSize
                attMask = [1] * seqLen + [0] * padSize

                yield textIds, attMask, textEmb

            if (loopForever == False):
                break

    f = lambda x, y=precision_settings: tf.convert_to_tensor(x, y)

    def _parse_function(textIds, attMask, textEmb):
        textIDs, att = f(textIds, tf.int32), f(attMask)
        tEmb = f(textEmb)
        return (textIDs, att), tEmb[0]

    dataset = tf.data.Dataset.from_generator(generatorFunc,
                                             output_types=(
                                                 tf.int32, precision_settings, precision_settings),
                                             output_shapes=(
                                                 (maxSeqLen,), (maxSeqLen,), encoderDims),
                                             )

    if (shuffleSize is not None):
        dataset = dataset.shuffle(shuffleSize)
    dataset = dataset.map(_parse_function).batch(batchSize)

    return dataset


def createTrainingAndValidationDataset(trainEmbeddings, valEmbeddings, batchSize, tokenizer, #targetCaptions,
                                       maxSeqLen=32, encoderDims=(1, 768),precision_16=False):
    

    logger.info(f"batchSize is {batchSize}")
    logger.info(f"maxSeqLen is {maxSeqLen}")
    logger.info(f"encoderDims is {encoderDims}")

    valDataset = createDataset(valEmbeddings, batchSize, tokenizer,
                               loopForever=False, maxSeqLen=maxSeqLen, encoderDims=encoderDims, precision_16=precision_16)


    trainDataset = createDataset(trainEmbeddings, batchSize, tokenizer,
                                 loopForever=True, maxSeqLen=maxSeqLen, encoderDims=encoderDims, precision_16=precision_16)

    return trainDataset, valDataset
