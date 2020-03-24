import random
import pandas as pd
import sys, os

class DoCleanInput:
    def do_cleaning(self, mappingfilepath, dataframefilepath, dataframe_sheet_name):

        if dataframefilepath.rpartition('.')[2] == 'csv':
            dataframe = pd.read_csv(dataframefilepath, encoding='utf-8', header=0, na_filter=False, error_bad_lines=False)
        elif dataframefilepath.rpartition('.')[2] == 'xlsx':
            dataframe = pd.read_excel(dataframefilepath, encoding='utf-8', header=0)
        elif dataframefilepath.rpartition('.')[2] == 'xls':
            dataframe = pd.read_excel(dataframefilepath, sheet_name=dataframe_sheet_name, encoding='utf-8', header=0)
        elif dataframefilepath.rpartition('.')[2] == 'xlsm':
            dataframe = pd.read_excel(dataframefilepath, sheet_name=dataframe_sheet_name, encoding='utf-8', header=0)
        else:
            print("file format not supported, please enter the file in .xlsx or .csv format.")

        # dataframe = dataframe.fillna('No_Value')

        dataframe.columns = dataframe.columns.astype(str)
        dataframe.rename(columns={list(dataframe)[0]: 'Initial_Value'}, inplace=True)

        mylist = ['False', 'No_Value', '0']
        pattern = '|'.join(mylist)
        deleted_col_list = []
        dataframe.columns = dataframe.columns.astype(str)
        for (columnName, columnData) in dataframe.iteritems():
            total_count = 0
            found_count = 0
            for values in columnData.values:
                if str(values) in pattern:
                    found_count += 1
                total_count += 1
            if found_count / total_count * 100 > 80:
                #print('Colunm Name : ', columnName)
                deleted_col_list.append(dataframe.columns.get_loc(columnName))

        dataframe.drop(dataframe.columns[deleted_col_list], axis=1, inplace=True)

        path_to_save = os.getcwd() + '/Data Sets/Cleaned_File.xlsx'
        dataframe.to_excel(path_to_save, index=False)
        return dataframe

def main():
    trans = DoCleanInput()
    trans.do_cleaning(mappingfilepath, dataframefilepath, dataframe_sheet_name)

if __name__ == '__main__':
    main()