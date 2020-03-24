import random
import pandas as pd
import sys, os

class DoCleanInput:
    def do_cleaning(self, mappingfilepath, dataframefilepath, dataframe_sheet_name):
        self.mappingfilepath = mappingfilepath
        self.dataframefilepath = dataframefilepath
        self.dataframe_sheet_name = dataframe_sheet_name
        #dataframe_sheet_name = 'Sheet1'
        #dataframefilepath = 'D:\shared\Machine Learning Tool\LDA
        # Files\Technip_HV_Induction_Motor_Data_Sheet_4370kW.xls'
        #mappingfilepath = 'D:\\shared\\Machine Learning Tool\\LDA Files\\Mapping_Technical_Data.xlsx'

        if (dataframefilepath.rpartition('.')[2] == 'csv'):
            dataframe = pd.read_csv(dataframefilepath, encoding='utf-8', header=0, na_filter=False,
                                     error_bad_lines=False)
        elif (dataframefilepath.rpartition('.')[2] == 'xlsx'):
            dataframe = pd.read_excel(dataframefilepath, sheet_name=dataframe_sheet_name, encoding='utf-8', header=0)
        elif (dataframefilepath.rpartition('.')[2] == 'xls'):
            dataframe = pd.read_excel(dataframefilepath, sheet_name=dataframe_sheet_name, encoding='utf-8', header=0)
        elif (dataframefilepath.rpartition('.')[2] == 'xlsm'):
            dataframe = pd.read_excel(dataframefilepath, sheet_name=dataframe_sheet_name, encoding='utf-8', header=0)
        else:
            print("file format not supported, please enter the file in .xlsx or .csv format.")

        mappingfile = pd.read_excel(mappingfilepath, encoding='utf-8', header=0)
        dataframe = dataframe.fillna('No_Value')

        mylist = ['False', 'No_Value', '0']
        pattern = '|'.join(mylist)
        total_count = 0
        found_count = 0
        deleted_col_list = []
        dataframe.columns = dataframe.columns.astype(str)

        '''Checking and deleting un-necessary values'''
        # for (columnName, columnData) in dataframe.iteritems():
        #     total_count = 0
        #     found_count = 0
        #     if dataframe.columns.get_loc(columnName) == 0:
        #         for values in columnData.values:
        #             for ind, rw in mappingfile.iterrows():
        #                 if str(values) == rw.Initial_Value:
        #                     found_count += 1
        #             total_count += 1
        #         if found_count / total_count * 100 < 50:
        #             deleted_col_list.append(dataframe.columns.get_loc(columnName))

        #dataframe.drop(dataframe.columns[deleted_col_list], axis=1, inplace=True)
        dataframe.rename(columns={list(dataframe)[0]: 'Initial_Value'}, inplace=True)
        #index_list = []
        # for index, row in dataframe.iterrows():
        #     flag = 0
        #     for indexmap, rowmap in mappingfile.iterrows():
        #         if str(row.Initial_Value) in rowmap.Initial_Value:
        #             flag = 1
        #             break
        #         elif rowmap.Initial_Value in str(row.Initial_Value):
        #             flag = 1
        #             break
        #         elif str(row.Initial_Value) in rowmap.Initial_Value:
        #             row.Initial_Value = row.Initial_Value.replace('(', '[')
        #             row.Initial_Value = row.Initial_Value.replace(')', ']')
        #             flag = 1
        #             break
        #         elif rowmap.Initial_Value in str(row.Initial_Value):
        #             row.Initial_Value = row.Initial_Value.replace('(', '[')
        #             row.Initial_Value = row.Initial_Value.replace(')', ']')
        #             flag = 1
        #             break
        #     if flag == 0:
        #         index_list.append(index)
        # dataframe = dataframe.drop(dataframe.index[index_list])
        # dataframe.reset_index(drop=True, inplace=True)

        mylist = ['False', 'No_Value', '0']
        pattern = '|'.join(mylist)
        total_count = 0
        found_count = 0
        deleted_col_list = []
        dataframe.columns = dataframe.columns.astype(str)
        #dataframe.to_excel('D:\\shared\\Machine Learning Tool\\LDA Files\\testdatatransfinal.xlsx', index=False)
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

        #path_to_save = 'D:\\shared\\Machine Learning Tool\\LDA Files\\test files\\testlvl3.xlsx'
        #dataframe.to_excel(path_to_save, index=False)

        #dataframe.to_excel('D:\\shared\\Machine Learning Tool\\LDA Files\\test files\\testlvl1.xlsx', index=False)
        path_to_save = os.getcwd() + '/Data Sets/Cleaned_File.xlsx'
        dataframe.to_excel(path_to_save, index=False)
        return dataframe

def main():
    trans = DoCleanInput()
    trans.do_cleaning(mappingfilepath, dataframefilepath, dataframe_sheet_name)

if __name__ == '__main__':
    main()