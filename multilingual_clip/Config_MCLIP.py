import transformers


class MCLIPConfig(transformers.PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(self, modelBase='aubmindlab/bert-base-arabertv2', transformerDimSize=1024, imageDimSize=512, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)
