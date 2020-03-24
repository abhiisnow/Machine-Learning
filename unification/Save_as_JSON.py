"""Import all required libraries"""
import pandas as pd
import json
import re
import os

"""Class to save the excel data as JSON"""

class Save_as_JSON:
    """Method to save in JSON fromat"""
    def saving_json(self, dataframefilepath, jsonframefilepath, input_file_name_folder_for_columns):
        """Reading required 2 excel files into dataframes"""
        dataframe = pd.read_excel(dataframefilepath, encoding='utf-8', header=0)
        jsonframe = pd.read_excel(jsonframefilepath, encoding='utf-8', header=0)
        '''Deleting extra column'''
        del dataframe['Initial_Value']
        dataframe = dataframe[~dataframe.Selected_Prediction.str.contains("no input")]
        # out = dataframe.to_json(orient='records')[1:-1].replace('}],[{', '},{')
        # with open(r'D:\\shared\\Machine Learning Tool\\LDA Files\\test files\\Export_DataFrame.json', 'w') as f:
        #     f.write(out)
        '''Resetting index after drop'''
        dataframe.reset_index(drop=True, inplace=True)
        lstdel = []
        '''deleting and reindexing after removing extra values which are not required for JSON'''
        for insav, rwsave in dataframe.iterrows():
            flag = 0
            for injson, rwjson in jsonframe.iterrows():
                if rwsave.Selected_Prediction in rwjson.Json_Label:
                    flag = 1
            if flag == 0:
                lstdel.append(insav)
        dataframe = dataframe.drop(dataframe.index[lstdel])
        dataframe.reset_index(drop=True, inplace=True)

        df_to_save = dataframe[['Selected_Prediction']].copy()

        '''Saving only the unique values in the dataframe'''
        dataframe.drop(columns=['Selected_Prediction'], inplace=True)
        dataframe = dataframe.T.drop_duplicates().T
        dataframe.insert(0, 'Selected_Prediction', df_to_save.Selected_Prediction)

        path_to_save = os.getcwd() + '\\Data Sets\\Combined_all_column_output\\' + input_file_name_folder_for_columns \
                       + '\\Prediction_with_Unique_Orginal_Values.xlsx'
        dataframe.to_excel(path_to_save, index=False)

        '''Adding missing fields required for JSON file'''
        lstadd = []
        for injson, rwjson in jsonframe.iterrows():
            flag = 0
            for insav, rwsave in df_to_save.iterrows():
                if rwjson.Json_Label in rwsave.Selected_Prediction:
                    flag = 1
            if flag == 0:
                lstadd.append(rwjson.Json_Label)
        for item in lstadd:
            df_to_save = df_to_save.append({'Selected_Prediction': item}, ignore_index=True)

        # print(dataframe)
        '''Adding to list to finally save it as JSON'''
        df = []
        for (columnName, columnData) in dataframe.iteritems():
            if dataframe.columns.get_loc(columnName) > 0:
                for indata, rwdata in dataframe.iterrows():
                    for insav, rwsave in df_to_save.iterrows():
                        if rwdata.Selected_Prediction == rwsave.Selected_Prediction:

                            df_to_save.loc[insav, 'Value_to_Save'] = rwdata[dataframe.columns.get_loc(columnName)]
                            # print(rwdata[dataframe.columns.get_loc(columnName)])
                df.append(df_to_save.set_index('Selected_Prediction').T.to_dict('record'))

        '''Cleaning list before saving it as JSON'''
        df = json.dumps(df).replace('\\\"', '')
        df = json.dumps(df).replace('}], [{', '},{')
        df = json.dumps(df).replace('"[[', '{"Results": [')
        df = json.dumps(df).replace(']]', ']}')
        df = re.sub(r"[\\]", '', df)
        df = re.sub(r"[\\\"]", '\'', df)
        df = re.sub(r"NaN", '\'\'', df)
        df = re.sub(r'^(..)', "", df)
        df = re.sub(r'(...)$', "", df)
        df = eval(df)

        '''Saving in JSON format'''
        path_to_save = os.getcwd() + '\\Data Sets\\Combined_all_column_output\\' + input_file_name_folder_for_columns\
                       + '\\' + input_file_name_folder_for_columns + '.json'

        with open(path_to_save, 'w') as json_file:
            json.dump(df, json_file)


def main():
    saves = Save_as_JSON()
    input_file_name_folder_for_columns = 'Kopie von Motor Enquiry (2).xlsx'
    dataframefilepath = os.getcwd() + '\\Data Sets\\Combined_all_column_output\\' + input_file_name_folder_for_columns \
                        + '\\Prediction_with_Orginal_Values.xlsx'
    jsonframefilepath = os.getcwd() + '\\LDA Files\\Json_Mapping file.xlsx'

    saves.saving_json(dataframefilepath, jsonframefilepath, input_file_name_folder_for_columns)


if __name__ == '__main__':
    main()
