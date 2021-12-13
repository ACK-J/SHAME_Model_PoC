import numpy as np
import os
import sys
from colorama import init
init(strip=not sys.stdout.isatty())
from termcolor import cprint
from pyfiglet import figlet_format
from tensorflow.keras.models import Model, Sequential
import time

from run import build_deep_fingerprinting_model, build_ensemble_model, preprocessor, feature_extraction

if __name__ == '__main__':
    def runcommand(cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        std_out, std_err = proc.communicate()
        return proc.returncode, std_out, std_err

    os.system("./demo.sh")

    pcap = "./predict"
    MAX_SIZE = 1000
    numpy2dSizeArray, numpy2dTimeArray, _, NUM_CLASSES = preprocessor(pcap, MAX_SIZE)
    numpy2dSizeArray = numpy2dSizeArray[..., np.newaxis]
    numpy2dTimeArray = numpy2dTimeArray[..., np.newaxis]

    INPUT_SHAPE = (MAX_SIZE, 1)
    NUM_CLASSES = 20
    # Create two DF models
    model = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)
    model2 = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)


    #model.load_weights("./Google/time_weights/time-model-057-0.996356-0.994062.h5")
    #model2.load_weights("./Google/size_weights/size-model-068-0.997594-0.997990.h5")

    # model.load_weights("./alexa/time_weights/time-model-041-0.886523-0.864234.h5")
    # model2.load_weights("./alexa/size_weights/size-model-035-0.932176-0.919757.h5")

    model.load_weights("./20_question_alexa/time-model-057-0.862883-0.861824.h5")
    model2.load_weights("./20_question_alexa/size-model-067-0.902822-0.862916.h5")

    #  Make sure to not train either model any further
    model.trainable = False
    model2.trainable = False

    START = time.time()
    #print("Getting Flatten layer using the time array")
    print("Running network traffic through the SHAME model")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for time
    flatten_model1 = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    outputs1 = flatten_model1.predict(numpy2dTimeArray, verbose=1)  # (N, 1024)

    #print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model2 = Model(inputs=model2.input, outputs=model2.get_layer('flatten').output)
    outputs2 = flatten_model2.predict(numpy2dSizeArray, verbose=1)  # (N, 1024)

    #  Combine the two models outputs, just created
    ensemble_input = np.concatenate((outputs1, outputs2), axis=1)  # (N, 2048) (samples, combined flattened layer)
    ensemble_model = build_ensemble_model(ensemble_input.shape, NUM_CLASSES)

    #ensemble_model.load_weights("./Google/ensemble_weights/ensemble-model-096-0.999033-0.999208.h5")
    #ensemble_model.load_weights("./alexa/ensemble_weights/ensemble-model-001-0.951374-0.956835.h5")
    ensemble_model.load_weights("./20_question_alexa/ensemble-model-002-0.960283-0.927167.h5")


    predictions = ensemble_model.predict(ensemble_input)
    classes = ["do dogs dream?", "how hot is the sun?", "what is the price of monero?", "what time is it?", "how tall is the empire state building?", "give me a random fact.", "how many days are in a year?", "how far away is the moon?", "what is the stock price of tesla?", "what is the price of silver?", "how is the dow jones doing?", "tell me a joke.", "what is the price of gold?", "what is the capital of spain?", "what is the fastest animal in the world?", "what is the weather is Canberra, Australia?", "what is the price of bitcoin?", "is a tomato a fruit or a vegetable?", "how many days are in september?", "how deep is the indian ocean?"]
    STOP = time.time()
    prediction = np.argmax(predictions, axis=1)[0]
    #print(prediction)
    cprint(figlet_format(classes[prediction], font='starwars', width=100), 'green', attrs=['bold'])

    #print(STOP - START)
