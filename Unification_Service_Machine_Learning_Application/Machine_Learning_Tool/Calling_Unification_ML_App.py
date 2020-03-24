import pandas as pd
from termcolor import colored
import numpy as np
import os, sys
from Machine_Learning_Tool_Unification_Service_with_Pass_Parameter import Process_All_Text_Columns

def all_inputs():
    input_file_name = sys.argv[1]
    input_file_sheet_name = sys.argv[2]
    return input_file_name, input_file_sheet_name

def main():
    input_file_name, input_file_sheet_name = all_inputs()
    mlunification = Process_All_Text_Columns()
    print('input_file_name: ', input_file_name, 'input_file_sheet_name: ', input_file_sheet_name)
    mlunification.ml_process_general_text_columns_processing(input_file_name, input_file_sheet_name)
    print(colored('Application executed successfully..!!!', 'green'))

if __name__ == '__main__':
    main()