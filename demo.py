from kdd_pred.pipeline.batch_prediction import strat_batch_prediction


file_path = "/home/vinod/projects_1/KDD_END_2_END/data/KDDTrain_.csv"

if __name__ == "__main__":
    try:
        output = strat_batch_prediction(input_file_path=file_path)

    except Exception as e:
        print(e)