import tensorflow as tf
import numpy as np
import pickle
import datetime

def splitListIntoChunks(data, numChunks):
    chunkSize = int(len(data) / numChunks)
    chunks = []
    for i in range(numChunks - 1):
        start, end = i * chunkSize, (i + 1) * chunkSize
        chunks.append(data[start:end])

    chunks.append(data[end:])
    return chunks


def splitIntoValueChunks(data, numChunks, getValueFunc):
    values = [getValueFunc(d) for d in data]
    minValue, maxValue = np.min(values), np.max(values)
    chunkSize = (maxValue - minValue) / float(numChunks)

    data.sort(key=lambda x: getValueFunc(x))
    sizeCeil = minValue + chunkSize
    chunks, currentChunkIndex = [[]], 0
    for d in data:
        v = getValueFunc(d)
        while (v > sizeCeil):
            chunks.append([])
            sizeCeil += chunkSize
            currentChunkIndex += 1
        chunks[currentChunkIndex].append(d)

    return chunks


def startGraphLogging():
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/func/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    return writer, logdir


def finishGraphLogging(writer, logdir):
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)


class CustomSaveCallBack(tf.keras.callbacks.Callback):

    def __init__(self, saveName, saveInterval=10, firstSavePoint=-1,log_name=" ",tokenizer=None,model=None):
        super().__init__()
        self.saveName = saveName
        self.saveInterval = saveInterval
        self.firstSavePoint = saveInterval if firstSavePoint < 0 else firstSavePoint
        self.saveCounter = 1
        self.log_name = log_name
        self.tokenizer = tokenizer
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        print()
        print("on_epoch_end saving, epoch num is: ", epoch +1 )
        if (epoch + 1 >= self.firstSavePoint):
            print("(epoch + 1 >= self.firstSavePoint): ", epoch + 1)
            
            print("on_epoch_end saving, saveCounter num is: ", self.saveCounter )
            if (self.saveCounter % self.saveInterval == 0):
                
                main_dir = "/home/lenovo/Desktop/arabic_clip/arabert_v2_vit_B_16_plus/"
                print("Saving model as {} from keras callback!".format(main_dir + self.saveName.format(epoch + 1) + "_internal_" + '.keras'))
                print(self.saveName + " " + str(epoch + 1))
                # self.model.save_weights(self.saveName.format(epoch + 1)) #  + '.h5')
                self.model.save(main_dir + self.saveName + " " + str(epoch + 1) + "_" + datetime.datetime.now().strftime("%Y %m %d - %H %M %S") + "_epoch_" + str(epoch + 1) + "_internal_" + '.keras')

                print("Saving model ......................")
                saveNameBase = self.log_name + datetime.datetime.now().strftime("%Y %m %d - %H %M %S")

                # dense_weights = dense_layer.get_weights()
                # Access the weights of the postTransformation dense layer using TensorFlow graph
                # graph = tf.compat.v1.get_default_graph()
                # dense_weights = graph.get_tensor_by_name('tf_bert_model/postTransformation/kernel:0')

                self.tokenizer.save_pretrained(main_dir + saveNameBase + '-Tokenizer-after-finish-training' + "epoch" + str(epoch + 1))
                self.model.transformer.save_pretrained(main_dir + saveNameBase + '-Transformer-after-finish-training' + "epoch" + str(epoch + 1))

                # Save the layer using pickle
                print("Saving the pickle file")
                
                pickle_file_path = main_dir + 'heads_of_the_model_' + self.saveName +  str(epoch + 1)+ "_.pickle"
                print("pickle file name: ", pickle_file_path)
                with open(pickle_file_path, 'wb') as pickle_file:
                    pickle.dump(self.model.postTransformation.get_weights(), pickle_file)

            self.saveCounter += 1


def saveTokenizer(base='gpt2', dumpPath='GPT2-Tokenizer.pkl'):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    with open(dumpPath, 'wb') as fp:
        pickle.dump(tokenizer, fp)


def loadTokenizer(dumpPath='GPT2-Tokenizer.pkl'):
    with open(dumpPath, 'rb') as fp:
        return pickle.load(fp)
