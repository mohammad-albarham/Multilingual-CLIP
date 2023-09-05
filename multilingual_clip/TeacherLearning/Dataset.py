import tensorflow as tf
from tqdm import tqdm

def createDataset(embeddings, batchSize, tokenizer, maxSeqLen=32, loopForever=True,
                  shuffleSize=None, encoderDims=(1, 768)):
    # print("Start createDataset")

    # total_num = len(embeddings)

    def generatorFunc():
        while True:
            # print("Start while")
            num = 0
            embeddings.shuffle()

            # print("Finishing shuffle")

            # print("Sample of embeddings: ", embeddings[0])
            # print("len(embeddings): ", len(embeddings))

            # Looping over all embeddings on the dataset

            print("Start looping on Full data")

            for d in embeddings:

                # print("Inside for loop shuffle")

                textEmb,caption = d['embedding'], d['ar_caption']
                # try:
                # print("Inside the try: ")

                # start_with_ar = targetCaptions.filter(lambda example: example["id"]==key)

                # caption = start_with_ar['ar_caption'] # targetCaptions[key]['ar_caption'] #  caption_multi

                # print("key: ", key)
                # print("caption: ", caption)

                if (caption is None):
                    continue

                textIds = tokenizer.encode(caption[0])
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
                # except:
                #     print("Inside the except: ")
                #     pass

            if (loopForever == False):
                break

            print("Finish looping on Full data")
        
        # print("End createDataset")

    # print("Decalre f: ")
    f = lambda x, y = tf.float32: tf.convert_to_tensor(x, y)

    def _parse_function(textIds, attMask, textEmb):
        textIDs, att = f(textIds, tf.int32), f(attMask)
        tEmb = f(textEmb)
        return (textIDs, att), tEmb[0]

    # print("_parse_function")

    dataset = tf.data.Dataset.from_generator(generatorFunc,
                                             output_types=(
                                                 tf.int32, tf.float32, tf.float32),
                                             output_shapes=(
                                                 (maxSeqLen,), (maxSeqLen,), encoderDims),
                                             )
    # dataset
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

    # print("Return dataset")
    return dataset


def createTrainingAndValidationDataset(trainEmbeddings, valEmbeddings, batchSize, tokenizer,
                                       maxSeqLen=32, encoderDims=(1, 768)):
    
    print("validation set loading, start: .....")
    valDataset = createDataset(valEmbeddings, batchSize, tokenizer,
                               loopForever=False, maxSeqLen=maxSeqLen, encoderDims=encoderDims)
    print("validation set loading, end: .....")

    print("training set loading, start: .....")
    trainDataset = createDataset(trainEmbeddings, batchSize, tokenizer,
                                 loopForever=True, maxSeqLen=maxSeqLen, encoderDims=encoderDims)
    
    print("training set loading, end: .....")

    # print("Return trainDataset: ", trainDataset)
    # print("Return valDataset: ", valDataset)

    return trainDataset, valDataset
