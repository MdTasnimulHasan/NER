from simpletransformers.ner import NERModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel, NERArgs
import torch
import argparse
import os


def get_args():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(
        prog='Named Entitity Recognition',
        description='Extracting Named Entity from text',
        epilog='Example: python extract_named_entity.py filename ')
    
    parser.add_argument('filename', type = str, default = 'GB00BD9MP466/prospectus_investment_ovjective.txt', help='source txt file')
    parser.add_argument('-smd','--model_path', type = str, default = 'E:/Tasnim/SonyCSL/NER_model.pt', help='path of saved weights of Named Entity model')
    parser.add_argument('-r', '--remove_input_file', action='store_true')
    parser.add_argument('-c', '--cleanse', action='append', default = ['\t','\n'])


    
    return parser.parse_args()


def cleanse_text(args):
    
    
    basedir = os.path.dirname(args.filename)
    basename = os.path.basename(args.filename)

    uncleansed_txt_filepath =  os.path.join(os.path.split(__file__)[0], basedir, basename)

    """ Read the input file. """
    with open(uncleansed_txt_filepath, 'r') as fi:
        data = fi.read()  # Read data is whole contents, not line by line.

    """ Cleanse the data. """
    output = []

    # Cleanse charactor that is provided as arguments.
    cleansed_text = data
    for cleansed_charactor in args.cleanse:  # args.cleanse is a list of characters.
        cleansed_text = cleansed_text.replace(cleansed_charactor, ' ')

        # Convert multiple spaces to a single space.
        output.append(' '.join(cleansed_text.split()))

    """ Write the output file. """
    output_file = os.path.join(basedir, 'cleansed_' + basename)
    with open(output_file, 'w') as fo:
        for cleansed_text in output:
            fo.write(cleansed_text+'\n')

    """ Remove the input file. """
    if args.remove_input_file:
        os.remove(args.filename)

    """ Print out the output file as return value. """
    print(output_file)
    return output_file
    
def extract_sentences_from_file(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into sentences using period as the delimiter
            line_sentences = line.strip().split('.')
            # Extend the sentences list with non-empty sentences
            sentences.extend(sentence.strip() for sentence in line_sentences if sentence.strip())
    return sentences



def extract_entities_from_prediction(args, prediction):
    
    exctracted_named_entities = []
    for i in range (0, len(prediction)):
        prediction_length = len(prediction[i])
        for idx in range(0,prediction_length,1):
          kv_pair = prediction[i][idx]
          for key in kv_pair:
            if kv_pair[key] != 'O':
              exctracted_named_entities.append(key)
              
    refined_named_entities = list(set(exctracted_named_entities))
    
    
    basedir = os.path.dirname(args.filename)
    basename = os.path.basename(args.filename)
    
    ner_savefile_name = "ner_" +  basename
    ner_savepath = os.path.join(os.path.split(__file__)[0], basedir, ner_savefile_name)
    
    with open(ner_savepath, 'w') as file:
        for item in refined_named_entities:
            file.write(str(item) + '\n')
    
    
    return refined_named_entities

def main():
    
    args = get_args()
    
    cleansed_txt_file_path = cleanse_text(args)
    
    sentences = extract_sentences_from_file(cleansed_txt_file_path)
    
    model = torch.load(args.model_path)

    model.eval()
    
    prediction, model_output = model.predict(sentences)
    
    named_entities  = extract_entities_from_prediction(args, prediction)
    print(named_entities)
    
    
    return
    
if __name__ == '__main__':
    main()


