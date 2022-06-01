from src.utils import write_conll
import os

def main_train(model,train_directory,validation_directory):
    for epoch in range(30):
        print('Epoch number: ', epoch + 1)
        model.Train(train_directory)

        # Predicting using the validation data and saving the current model
        predictions = model.Predict(validation_directory)
        write_conll(f"/Users/yaseminsavas/turkish-parser/results/dev_epoch_{epoch + 1}.conllu",
                    predictions)
        print("Predictions are printed.")

        # Evaluation with the validation data
        path = f"/Users/yaseminsavas/turkish-parser/results/dev_epoch_{epoch + 1}.conllu"
        os.system(
            'python /Users/yaseminsavas/turkish-parser/src/evaluation_script/conll17_ud_eval.py -v -w /Users/yaseminsavas/turkish-parser/src/evaluation_script/weights.clas '
            + validation_directory + ' ' + path + ' > ' + path + '.txt')

        print("Performance is evaluated.")
        print(" ")