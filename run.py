#!/usr/bin/env python3
import numpy as np
from sys import exit
from os.path import dirname, isdir, exists
from os import walk, makedirs
from scapy.layers.inet import IP
from scapy.utils import PcapReader
from argparse import ArgumentParser
from sklearn.utils import shuffle
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.backend import concatenate
from keras.layers import Input, Concatenate, Dense, BatchNormalization
from keras.layers.core import Activation, Dense, Dropout
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from pickle import load, dump
from df import DFNet

"""
Description:    When this script runs, it will recursively search through directories to find
                PCAP files. For each PCAP file, parse through grabbing timestamps and 
                checking if packet is outgoing or incoming.
                After PCAP is fully parsed, output to a text file (or otherwise specified)
                Continue with the next PCAP until no other PCAP.
                Should combine all the txt files into one directory
Date:           4/26/2021
Usage:          python3 run.py --input-data ./commands
                python3 run.py --load-time ./time_dump.pkl --load-size ./size_dump.pkl --load-classes ./y.pkl
                python3 run.py --load-weights-time ./time_weights/time-model-050-0.708984-0.727088.h5 
                               --load-weights-size ./size_weights/size-model-040-0.720978-0.672098.h5
                python3 run.py --load-time ./time_dump.pkl --load-size ./size_dump.pkl --load-classes ./y.pkl
                               --load-weights-time ./time_weights/time-model-050-0.708984-0.727088.h5 
                               --load-weights-size ./size_weights/size-model-040-0.720978-0.672098.h5
"""

IP_TARGET = ["10.150.101.1", "10.150.101.2"] #"10.150.101.1", "10.150.101.2"]#"10.63.1.88"]#"192.168.1.2", "192.168.128.2", "10.63.1.144"]  # The IP address of the smart home device


def save_to_file(sequence, path, delimiter='\t'):
    """save a packet sequence (2-tuple of time and direction) to a file"""
    if not exists(dirname(path)):
        makedirs(dirname(path))
    with open(path, 'w') as file:
        for packet in sequence:
            line = '{t}{b}{d}\n'.format(t=packet[0], b=delimiter, d=packet[1])
            file.write(line)


def preprocessor(inpath, MAX_SIZE):
    """
    :param input: root directory path containing pcap files
    :return: N/A
    """

    #print("Processing all pcaps in the " + str(inpath) + " directory...")
    print("Analyzing network traffic of smart assistant")
    # create list of pcap files to process
    flist = []
    for root, dirs, files in walk(inpath):
        # filter for only pcap files
        files = [fi for fi in files if fi.endswith(".pcap")]
        flist.extend([(root, f) for f in files])

    return feature_extraction(flist, MAX_SIZE)


def feature_extraction(flist, MAX_SIZE):
    """
    :param flist: array of tuples {<path to pcap question directory>, <pcap output file name>}
    :return: numpy2dSizeArray, numpy2dTimeArray
    """
    # initialize two numpy arrays that will hold all the data
    numpy2dSizeArray = np.empty((0, MAX_SIZE), float)
    numpy2dTimeArray = np.empty((0, MAX_SIZE), float)
    y = list()
    label_index = 0
    folders_seen = {}
    #  Go through each pcap file = ( path to dir, filename)
    for file in flist:
        if file[0] not in folders_seen:
            folders_seen[file[0]] = label_index
            #print("Loaded class #" + str(label_index) + "...")
            label_index += 1
        #  Open the pcap
        with PcapReader(file[0] + "/" + file[1]) as packets:
            packetCount = 1
            sizeArr = []  # size * direction
            timeArr = []  # time * direction
            for packet in packets:
                if packetCount > MAX_SIZE:
                    break
                direction = get_packetDirection(packet)
                size = get_packetSize(packet)
                if packetCount == 1:
                    startTime = get_packetTime(packet)
                    packetCount += 1
                else:
                    endTime = get_packetTime(packet)
                    # ensure all the values (size, direction, and time) for each packet exists before adding to numpy array
                    if direction != 0 and size != 0 and endTime != 0:
                        time = endTime - startTime
                        time = float(time) * 1000  # Converting to ms
                        startTime = endTime
                        sizeArr.append(size * direction)
                        timeArr.append(time * direction)
            if sizeArr == [] or timeArr == []:
                continue
            y.append(folders_seen[file[0]])
            # Padding
            while len(sizeArr) < MAX_SIZE:
                sizeArr.append(0)
            while len(timeArr) < MAX_SIZE:
                timeArr.append(0)
            if len(sizeArr) > MAX_SIZE:
                sizeArr = sizeArr[:MAX_SIZE]
            if len(timeArr) > MAX_SIZE:
                timeArr = timeArr[:MAX_SIZE]

            numpySizeArr = np.asarray(sizeArr)
            numpyTimeArr = np.asarray(timeArr)

            numpy2dSizeArray = np.vstack((numpy2dSizeArray, numpySizeArr))
            numpy2dTimeArray = np.vstack((numpy2dTimeArray, numpyTimeArr))

    numpy2dSizeArray, numpy2dTimeArray, y = shuffle(numpy2dSizeArray, numpy2dTimeArray, np.array(y), random_state=100)
    return numpy2dSizeArray, numpy2dTimeArray, y, label_index


def get_packetDirection(packet):
    try:
        if packet[IP].dst in IP_TARGET:  # packet coming to smart home device
            return -1
        elif packet[IP].src in IP_TARGET:  # packet going from smart home device
            return 1
        return 0  # Error
    except IndexError as e:
        return 0


def get_packetSize(packet):
    try:
        if packet[IP].len is not None:
            return packet[IP].len
        return 0  # Error
    except IndexError as e:
        return 0


def get_packetTime(packet):
    time = packet.time
    return time


def check_args(args):
    if args.input_data is None and args.load_time is None and args.load_size is None and args.load_classes is None:
        exit('Please run the program using either --input-data or --load-time, --load-size, and --load-classes')
    if args.load_weights_time is not None and args.load_weights_size is None:
        exit('Please provide filepaths to weights for both --load-weights-time and --load-weights-size')
    elif args.load_weights_time is None and args.load_weights_size is not None:
        exit('Please provide filepaths to weights for both --load-weights-time and --load-weights-size')


def build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES):
    model = DFNet.build(INPUT_SHAPE, NUM_CLASSES)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def build_ensemble_model(shape, NUM_CLASSES):
    """
    This ensemble model is based on the Deep Fingerprinting Model's flatten and dense layers
    before classification.
    :param shape: The shape of the ensembled training data
    :return: The ensemble model
    """
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Activation('relu', name='fc1_act'))
    model.add(Dropout(0.5, name='fc1_dropout'))
    model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Activation('relu', name='fc2_act'))
    model.add(Dropout(0.7, name='fc2_dropout'))
    model.add(Dense(NUM_CLASSES, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
    model.add(Activation('softmax', name="softmax"))

    adam = Adam(lr=0.01)
    sgd = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def train_SHAME_Model(numpy2dTimeArray, numpy2dSizeArray, INPUT_SHAPE, NUM_CLASSES, time_weights, size_weights, y):
    if not isdir("./time_weights"):
        makedirs("./time_weights")
    if not isdir("./size_weights"):
        makedirs("./size_weights")
    if not isdir("./ensemble_weights"):
        makedirs("./ensemble_weights")
    if not isdir("./pics"):
        makedirs("./pics")

    checkpoint1 = ModelCheckpoint('./time_weights/time-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_loss',
                                  save_best_only=True, mode='auto')
    checkpoint2 = ModelCheckpoint('./size_weights/size-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_loss',
                                  save_best_only=True, mode='auto')
    checkpoint3 = ModelCheckpoint('./ensemble_weights/ensemble-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_loss',
                                  save_best_only=True, mode='auto')

    tensorboard_callback = TensorBoard(log_dir="./logs")
    # Create two DF models
    model = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)
    model2 = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)

    #  Check to see if pre-trained weights were passed or not
    if time_weights is None or size_weights is None:
        df_time_history = model.fit(numpy2dTimeArray, to_categorical(y),
                                    batch_size=256,
                                    epochs=70,
                                    validation_split=0.10,
                                    verbose=True,
                                    callbacks=[checkpoint1])
        df_size_history = model2.fit(numpy2dSizeArray, to_categorical(y),
                                     batch_size=256,
                                     epochs=70,
                                     validation_split=0.10,
                                     verbose=True,
                                     callbacks=[checkpoint2])
    else:  # Load weights
        model.load_weights(time_weights)
        model2.load_weights(size_weights)

    #  Make sure to not train either model any further
    model.trainable = False
    model2.trainable = False
    from keras.utils import plot_model
    plot_model(model, to_file='./pics/time_model.png', show_shapes='True')
    plot_model(model2, to_file='./pics/size_model2.png', show_shapes='True')

    print("Getting Flatten layer using the time array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for time
    flatten_model1 = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    outputs1 = flatten_model1.predict(numpy2dTimeArray, verbose=1)  # (N, 1024)

    print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model2 = Model(inputs=model2.input, outputs=model2.get_layer('flatten').output)
    outputs2 = flatten_model2.predict(numpy2dSizeArray, verbose=1)  # (N, 1024)

    #  Combine the two models outputs, just created
    ensemble_input = np.concatenate((outputs1, outputs2), axis=1)  # (N, 2048) (samples, combined flattened layer)
    model3 = build_ensemble_model(ensemble_input.shape, NUM_CLASSES)

    model3.fit(x=ensemble_input, y=to_categorical(y),
               batch_size=256,
               epochs=300,
               validation_split=0.15,
               verbose=True,
               callbacks=[checkpoint3])
    plot_model(model3, to_file='./pics/ensemble_model3.png', show_shapes='True')


def parse_arguments():
    """parse command-line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--input-data", metavar=' ', help='filepath of folders containing pcaps')
    parser.add_argument("--load-time", metavar=' ',
                        help='filepath containing pre-saved data for training time * direction')
    parser.add_argument("--load-size", metavar=' ',
                        help='filepath containing pre-saved data for training size * direction')
    parser.add_argument("--load-classes", metavar=' ', help='filepath containing pre-saved data for training (classes)')
    parser.add_argument("--load-weights-time", metavar=' ', help='filepath to time weights to load into model')
    parser.add_argument("--load-weights-size", metavar=' ', help='filepath to size weights to load into model')
    parser.add_argument("--DeepVC", metavar=' ', help='Specifying to use the DeepVC Fingerprinting model instead of the SHAME model')
    args = parser.parse_args()
    check_args(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    NUM_CLASSES = 20  # Default  MAKE THIS INTO A COMMAND LINE OPTION
    MAX_SIZE = 1000  # Default

    if args.input_data is not None:
        # PROCESS DATA
        numpy2dSizeArray, numpy2dTimeArray, y, NUM_CLASSES = preprocessor(args.input_data, MAX_SIZE)
        print("Number of classes " + str(NUM_CLASSES))
        numpy2dSizeArray = numpy2dSizeArray[..., np.newaxis]
        numpy2dTimeArray = numpy2dTimeArray[..., np.newaxis]
        # SAVE PROCESSED DATA
        with open("size_dump.pkl", "wb") as fp:
            dump(numpy2dSizeArray, fp)
        with open("time_dump.pkl", "wb") as fp:
            dump(numpy2dTimeArray, fp)
        with open("y.pkl", "wb") as fp:
            dump(y, fp)
    else:  # not given input data
        if args.load_time is None and args.load_size is None and args.load_classes is None:
            exit('Please provide values for all three arguments (--load-time, --load-size, --load-classes) when loading in your own data')
        else:
            # LOAD PROCESSED DATA
            with open(args.load_time, "rb") as fp:
                numpy2dTimeArray = load(fp)
            with open(args.load_size, "rb") as fp:
                numpy2dSizeArray = load(fp)
            with open(args.load_classes, "rb") as fp:
                y = load(fp)

    INPUT_SHAPE = (MAX_SIZE, 1)

    if args.DeepVC is not None:
        # USE https://arxiv.org/abs/2005.09800
        from DeepVC import CNN
        cnn = CNN()
        modelPath = cnn.train(numpy2dSizeArray, to_categorical(y), NUM_CLASSES)
        modelPath = cnn.train(numpy2dTimeArray, to_categorical(y), NUM_CLASSES)
    else:
        #  Check if pre-trained weights were passed
        if args.load_weights_time is not None and args.load_weights_size is not None:
            # USE Deep Fingerprinting with user weights for the models
            train_SHAME_Model(numpy2dTimeArray, numpy2dSizeArray, INPUT_SHAPE, NUM_CLASSES, args.load_weights_time, args.load_weights_size, y)
        else:
            # Train DF weights from scratch
            time_weights = None
            size_weights = None
            train_SHAME_Model(numpy2dTimeArray, numpy2dSizeArray, INPUT_SHAPE, NUM_CLASSES, time_weights, size_weights, y)
