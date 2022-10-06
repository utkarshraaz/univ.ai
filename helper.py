import tensorflow as tf


class Model:
    def __init__(self,inputDim,outputDim):
        self.inputDim=inputDim
        self.outputDim=outputDim

    def buildSmallModel(self):

        input= tf.keras.layers.Input((None, self.inputDim), name="input")
        reshape = tf.keras.layers.Reshape((-1, self.inputDim, 1), name="expand_dim")(input)
        cnn1= tf.keras.layers.Conv2D(filters=32,kernel_size=(11, 41),strides=(2, 2),padding="same",use_bias=False,name="conv_1")(reshape)
        bn1 = tf.keras.layers.BatchNormalization(name="conv_1_bn")(cnn1)
        relu1 = tf.keras.layers.ReLU(name="conv_1_relu")(bn1)

        cnn2= tf.keras.layers.Conv2D(filters=32,kernel_size=(11, 21),strides=(1, 2),padding="same",use_bias=False,name="conv_2")(relu1)
        bn2 = tf.keras.layers.BatchNormalization(name="conv_2_bn")(cnn2)
        relu2 = tf.keras.layers.ReLU(name="conv_2_relu")(bn2)
        reshape2 = tf.keras.layers.Reshape((-1, relu2.shape[-2] * relu2.shape[-1]))(relu2)



        gru1 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_1")
        bi1 = tf.keras.layers.Bidirectional(gru1, name=f"bidirectional_1", merge_mode="concat")(reshape2)
        dr1 = tf.keras.layers.Dropout(rate=0.5)(bi1)

        gru2 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_2")
        bi2 = tf.keras.layers.Bidirectional(gru2, name=f"bidirectional_2", merge_mode="concat")(dr1)
        dr2 = tf.keras.layers.Dropout(rate=0.5)(bi2)

        gru3 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_3")
        bi3 = tf.keras.layers.Bidirectional(gru3, name=f"bidirectional_3", merge_mode="concat")(dr2)
        dr3 = tf.keras.layers.Dropout(rate=0.5)(bi3)

        gru4 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_4")
        bi4 = tf.keras.layers.Bidirectional(gru4, name=f"bidirectional_4", merge_mode="concat")(dr3)
        dr4 = tf.keras.layers.Dropout(rate=0.5)(bi4)

        gru5 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_5")
        bi5 = tf.keras.layers.Bidirectional(gru5, name=f"bidirectional_5", merge_mode="concat")(dr4)

        x = tf.keras.layers.Dense(units=1024, name="dense1")(bi5)
        x = tf.keras.layers.ReLU(name="dense_1_relu")(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
    # Classification layer
        output = tf.keras.layers.Dense(units=self.outputDim + 1, activation="softmax")(x)

        return tf.keras.Model(input, output,name="DeepSpeech_2")


    def buildLargeModel(self):

        input= tf.keras.layers.Input((None, self.inputDim), name="input")
        reshape = tf.keras.layers.Reshape((-1, self.inputDim, 1), name="expand_dim")(input)
        cnn1= tf.keras.layers.Conv2D(filters=32,kernel_size=(11, 41),strides=(2, 2),padding="same",use_bias=False,name="conv_1")(reshape)
        bn1 = tf.keras.layers.BatchNormalization(name="conv_1_bn")(cnn1)
        relu1 = tf.keras.layers.ReLU(name="conv_1_relu")(bn1)

        cnn2= tf.keras.layers.Conv2D(filters=32,kernel_size=(11, 21),strides=(1, 2),padding="same",use_bias=False,name="conv_2")(relu1)
        bn2 = tf.keras.layers.BatchNormalization(name="conv_2_bn")(cnn2)
        relu2 = tf.keras.layers.ReLU(name="conv_2_relu")(bn2)

        cnn3= tf.keras.layers.Conv2D(filters=64,kernel_size=(11, 21),strides=(1, 2),padding="same",use_bias=False,name="conv_3")(relu2)
        bn3 = tf.keras.layers.BatchNormalization(name="conv_3_bn")(cnn3)
        relu3 = tf.keras.layers.ReLU(name="conv_3_relu")(bn3)

        reshape2 = tf.keras.layers.Reshape((-1, relu3.shape[-2] * relu3.shape[-1]))(relu3)



        gru1 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_1")
        bi1 = tf.keras.layers.Bidirectional(gru1, name=f"bidirectional_1", merge_mode="concat")(reshape2)
        dr1 = tf.keras.layers.Dropout(rate=0.5)(bi1)

        gru2 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_2")
        bi2 = tf.keras.layers.Bidirectional(gru2, name=f"bidirectional_2", merge_mode="concat")(dr1)
        dr2 = tf.keras.layers.Dropout(rate=0.5)(bi2)

        gru3 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_3")
        bi3 = tf.keras.layers.Bidirectional(gru3, name=f"bidirectional_3", merge_mode="concat")(dr2)
        dr3 = tf.keras.layers.Dropout(rate=0.5)(bi3)

        gru4 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_4")
        bi4 = tf.keras.layers.Bidirectional(gru4, name=f"bidirectional_4", merge_mode="concat")(dr3)
        dr4 = tf.keras.layers.Dropout(rate=0.5)(bi4)

        gru5 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_5")
        bi5 = tf.keras.layers.Bidirectional(gru5, name=f"bidirectional_5", merge_mode="concat")(dr4)
        dr5 = tf.keras.layers.Dropout(rate=0.5)(bi5)

        gru6 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_6")
        bi6 = tf.keras.layers.Bidirectional(gru6, name=f"bidirectional_6", merge_mode="concat")(dr5)
        dr6 = tf.keras.layers.Dropout(rate=0.5)(bi6)

        gru7 = tf.keras.layers.GRU(units=512,activation="tanh",recurrent_activation="sigmoid",use_bias=True,return_sequences=True,reset_after=True,name=f"gru_7")
        bi7 = tf.keras.layers.Bidirectional(gru6, name=f"bidirectional_7", merge_mode="concat")(dr6)



        x = tf.keras.layers.Dense(units=1024, name="dense1")(bi7)
        x = tf.keras.layers.ReLU(name="dense_1_relu")(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
    # Classification layer
        output = tf.keras.layers.Dense(units=self.outputDim + 1, activation="softmax")(x)

        return tf.keras.Model(input, output,name="DeepSpeech_2")

    def lossFunc(self,true, pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(true)[0], dtype="int32")
        input_length = tf.cast(tf.shape(pred)[1], dtype="int32")
        label_length = tf.cast(tf.shape(true)[1], dtype="int32")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int32")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int32")

        loss = tf.keras.backend.ctc_batch_cost(true, pred, input_length, label_length)
        return loss

    def getModel(self,size='large'):
        if size=='large':
            model=self.buildLargeModel()
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss=self.lossFunc)
            return model
        else:
            model=self.buildSmallModel()
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss=self.lossFunc)
            return model
