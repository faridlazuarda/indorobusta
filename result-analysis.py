import pandas as pd
import os
from icecream import ic
 
# exp_dataset.to_csv(os.getcwd() + r'/result/seed'+str(seed)+"/"+str(dataset)+"/"+str(exp_name)+".csv", index=False)

# Get the list of all files and directories
def result_summary(seed, dataset):
    path = str(os.getcwd()) + "/result/seed" + str(seed) + "/" + str(dataset) + "/"
    dir_list = os.listdir(path)
    result = pd.DataFrame()
    
    for file in dir_list:
        if ".csv" in file:
            df_first_row = pd.read_csv(path+file).iloc[0]
            parsed_filename = file[:-4].split("-")

            if "sr" in parsed_filename:
                record = {
                    "filename": file,
                    "model_target": parsed_filename[0],
                    "downstream_task": parsed_filename[1],
                    "perturbation_technique": parsed_filename[2],
                    "codemix_lang" : "id",
                    "perturb_ratio": parsed_filename[4],
                    "before_attack_acc": df_first_row["before_attack_acc"],
                    "after_attack_acc": df_first_row["after_attack_acc"],
                    "delta_acc": (df_first_row["before_attack_acc"] - df_first_row["after_attack_acc"]),
                    "avg_semantic_sim": df_first_row["avg_semantic_sim"],
                    "avg_running_time(s)": df_first_row["avg_running_time(s)"]
                }
            else:
                record = {
                    "filename": file,
                    "model_target": parsed_filename[0],
                    "downstream_task": parsed_filename[1],
                    "perturbation_technique": parsed_filename[2],
                    "codemix_lang" : parsed_filename[3],
                    "perturb_ratio": parsed_filename[5],
                    "before_attack_acc": df_first_row["before_attack_acc"],
                    "after_attack_acc": df_first_row["after_attack_acc"],
                    "delta_acc": (df_first_row["before_attack_acc"] - df_first_row["after_attack_acc"]),
                    "avg_semantic_sim": df_first_row["avg_semantic_sim"],
                    "avg_running_time(s)": df_first_row["avg_running_time(s)"]
                }

            df_dictionary = pd.DataFrame([record])
            result = pd.concat([result, df_dictionary], ignore_index=True)
            
    # result.to_csv("result-analysis/" + str() + ".csv")
    result = result.sort_values(by=['model_target', 'downstream_task', 'codemix_lang', 'perturb_ratio'])
    result.to_csv(os.getcwd() + r'/result-analysis/seed'+str(seed)+"-"+str(dataset)+".csv", index=False)

if __name__ == "__main__":
    result_summary(
        seed=26092020,
        dataset="valid"
    )
    
    result_summary(
        seed=24032022,
        dataset="valid"
    )
    
    result_summary(
        seed=42,
        dataset="valid"
    )