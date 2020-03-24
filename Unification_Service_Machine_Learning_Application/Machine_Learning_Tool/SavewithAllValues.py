from pandas import DataFrame
import pandas as pd
import json
import os
import numpy as np

class Save_With_All_Values:
    def saving_all_values(self, dataframefile, cleanedframefilepath, jsonframefilepath, input_file_name_folder_for_columns):
        self.dataframefile = dataframefile
        self.jsonframefilepath = jsonframefilepath
        self.cleanedframefilepath = cleanedframefilepath
        self.input_file_name_folder_for_columns = input_file_name_folder_for_columns

        #dataframefilepath = 'D:\shared\Machine Learning Tool\Data Sets\Combined_all_column_output\Kopie von Motor
        # Enquiry (2).xlsx\Predicted_Score_all_columns.xlsx'
        #cleanedframefilepath = 'D:\shared\Machine Learning Tool\Data Sets\Cleaned_File.xlsx'

        #jsonframefilepath = 'D:\shared\Machine Learning Tool\LDA Files\Json_Mapping file.xlsx'
        jsonframe = pd.read_excel(jsonframefilepath, encoding='utf-8', header=0)

        #dataframe = pd.read_excel(dataframefilepath, encoding='utf-8', header=0)
        dataframe = dataframefile.copy()
        dataframe = dataframe[['Initial_Value', 'Selected_Prediction']].copy()
        dataframe['Initial_Value_Copy'] = [word.lower() for word in dataframe['Initial_Value']]
        dataframe = dataframe.drop(dataframe[(dataframe['Initial_Value_Copy'] == '')].index)
        dataframe = dataframe.drop(dataframe[(dataframe['Initial_Value_Copy'] == ' ')].index)
        dataframe = dataframe.drop(dataframe[(dataframe['Initial_Value_Copy'] == 'no_value')].index)

        cleanedframe = pd.read_excel(cleanedframefilepath, encoding='utf-8', header=0)
        cleanedframe['Initial_Value_Copy'] = [str(word).lower() for word in cleanedframe['Initial_Value']]
        cleanedframe = cleanedframe.drop(cleanedframe[(cleanedframe['Initial_Value_Copy'] == '')].index)
        cleanedframe = cleanedframe.drop(cleanedframe[(cleanedframe['Initial_Value_Copy'] == ' ')].index)
        cleanedframe = cleanedframe.drop(cleanedframe[(cleanedframe['Initial_Value_Copy'] == 'no_value')].index)

        #print(cleanedframe.head())
        for (columnName, columnData) in cleanedframe.iteritems():
            for indataframe, rwdataframe in dataframe.iterrows():
                for incleanframe, rwcleanframe in cleanedframe.iterrows():

                    if rwdataframe.Initial_Value_Copy == rwcleanframe.Initial_Value_Copy:
                        dataframe.loc[indataframe, 'Initial_Value'] = rwcleanframe.Initial_Value
                        if rwcleanframe[cleanedframe.columns.get_loc(columnName)] == 'No_Value':
                            dataframe.loc[indataframe, columnName] = ''
                        else:
                            dataframe.loc[indataframe, columnName] = rwcleanframe[cleanedframe.columns.get_loc(columnName)]

        del dataframe['Initial_Value_Copy']

        for insav, rwsave in dataframe.iterrows():
            for injson, rwjson in jsonframe.iterrows():
                if rwsave.Selected_Prediction == rwjson.Machine_Label:
                    dataframe.loc[insav, 'Selected_Prediction'] = rwjson['Json_Label']

        path_to_save = os.getcwd() + '\\Data Sets\\Combined_all_column_output\\' + input_file_name_folder_for_columns\
                       + '\\Prediction_with_Orginal_Values.xlsx'
        dataframe.to_excel(path_to_save, index=False)


def main():
    saves = Save_With_All_Values()
    dataframefilepath = 'C:\\Users\\Z003XE7X\Documents\\Unification Service Machine Learning Application\\Machine ' \
                        'Learning Tool\\Data Sets\\Combined_all_column_output\\Kopie von Motor Enquiry (2).xlsx\\Predicted_Score_all_columns.xlsx'
    dataframefile = pd.read_excel(dataframefilepath, encoding='utf-8', header=0)
    cleanedframefilepath = os.getcwd() + '\\Data Sets\\Cleaned_File.xlsx'
    input_file_name_folder_for_columns = 'Kopie von Motor Enquiry (2).xlsx'
    jsonframefilepath = os.getcwd() + '\\Data Sets\\Json_Mapping file.xlsx'
    saves.saving_all_values(dataframefile, cleanedframefilepath, jsonframefilepath, input_file_name_folder_for_columns)

if __name__ == '__main__':
    main()