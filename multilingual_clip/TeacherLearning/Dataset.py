import tensorflow as tf


def createDataset(targetCaptions, embeddings, batchSize, tokenizer, maxSeqLen=32, loopForever=True,
                  shuffleSize=None, encoderDims=(1, 768)):
    def generatorFunc():
        while True:
            num = 0
            embeddings.shuffle()
            for d in embeddings:
                key, textEmb = d['id'], d['embedding']
                try:
                    caption = targetCaptions[key]['caption_multi']
                    if (caption is None):
                        continue

                    textIds = tokenizer.encode(caption)
                    seqLen = len(textIds)
                    if (seqLen > maxSeqLen):
                        continue

                    padSize = maxSeqLen - len(textIds)
                    textIds = textIds + [0] * padSize
                    attMask = [1] * seqLen + [0] * padSize
                    num = num + 1
                    # print("="*100)
                    # print("Number of examples", num )
                    # print("="*100)
                    yield textIds, attMask, textEmb
                except:
                    pass

            if (loopForever == False):
                break

    f = lambda x, y=tf.float32: tf.convert_to_tensor(x, y)

    def _parse_function(textIds, attMask, textEmb):
        textIDs, att = f(textIds, tf.int32), f(attMask)
        tEmb = f(textEmb)
        return (textIDs, att), tEmb[0]

    dataset = tf.data.Dataset.from_generator(generatorFunc,
                                             output_types=(
                                                 tf.int32, tf.float32, tf.float32),
                                             output_shapes=(
                                                 (maxSeqLen,), (maxSeqLen,), encoderDims),
                                             )
    dataset
    if (shuffleSize is not None):
        dataset = dataset.shuffle(shuffleSize)
    dataset = dataset.map(_parse_function).batch(batchSize)

    # print("="*100)

    # # Iterate through the dataset and count the elements
    # # dataset_length = sum(1 for _ in dataset)
    # # print("Dataset length:", tf.data.experimental.cardinality(dataset).numpy())
    # # print("Dataset length:", dataset_length)
    # print(dir(dataset))
    # # cardinality = tf.data.experimental.cardinality(dataset)
    # # print((cardinality == tf.data.experimental.INFINITE_CARDINALITY).numpy())
    # # print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())

    # # print("dataset: ", dataset)
    # print("="*100)

    return dataset


def createTrainingAndValidationDataset(trainEmbeddings, valEmbeddings, batchSize, tokenizer, targetCaptions,
                                       maxSeqLen=32, encoderDims=(1, 768)):
    valDataset = createDataset(targetCaptions, valEmbeddings, batchSize, tokenizer,
                               loopForever=False, maxSeqLen=maxSeqLen, encoderDims=encoderDims)
    trainDataset = createDataset(targetCaptions, trainEmbeddings, batchSize, tokenizer,
                                 loopForever=True, maxSeqLen=maxSeqLen, encoderDims=encoderDims)



    return trainDataset, valDataset
